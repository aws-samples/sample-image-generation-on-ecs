from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "runtime" / "compile"))
import _safety as compile_safety


class TestLimits:
    def test_default_values(self):
        lim = compile_safety.Limits()
        assert lim.min_ram_gb == 10.0
        assert lim.min_disk_gb == 20.0
        assert lim.max_workdir_gb == 60.0
        assert lim.interval_s == 10.0

    def test_custom_values(self):
        lim = compile_safety.Limits(
            min_ram_gb=4.0,
            min_disk_gb=5.0,
            max_workdir_gb=30.0,
            interval_s=2.0,
        )
        assert lim.min_ram_gb == 4.0
        assert lim.min_disk_gb == 5.0
        assert lim.max_workdir_gb == 30.0
        assert lim.interval_s == 2.0


class TestBytesFree:
    def test_returns_available_bytes(self):
        fake_stat = MagicMock()
        fake_stat.f_bavail = 1000
        fake_stat.f_frsize = 4096
        with patch("os.statvfs", return_value=fake_stat):
            result = compile_safety._bytes_free("/some/path")
        assert result == 1000 * 4096

    def test_returns_minus_one_on_oserror(self):
        with patch("os.statvfs", side_effect=OSError("no such path")):
            result = compile_safety._bytes_free("/nonexistent")
        assert result == -1


class TestMemAvailableBytes:
    def test_parses_proc_meminfo(self):
        meminfo_content = (
            "MemTotal:       32000000 kB\n"
            "MemFree:        10000000 kB\n"
            "MemAvailable:   20000000 kB\n"
            "Buffers:         1000000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=meminfo_content)):
            result = compile_safety._mem_available_bytes()
        assert result == 20000000 * 1024

    def test_returns_minus_one_on_oserror(self):
        with patch("builtins.open", side_effect=OSError("not linux")):
            result = compile_safety._mem_available_bytes()
        assert result == -1

    def test_returns_minus_one_when_memavailable_not_found(self):
        meminfo_content = "MemTotal:       32000000 kB\nMemFree:        10000000 kB\n"
        with patch("builtins.open", mock_open(read_data=meminfo_content)):
            result = compile_safety._mem_available_bytes()
        assert result == -1


class TestDirSize:
    def test_returns_zero_for_nonexistent(self, tmp_path):
        result = compile_safety._dir_size(tmp_path / "does_not_exist")
        assert result == 0

    def test_sums_file_sizes(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"x" * 100)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_bytes(b"y" * 200)
        result = compile_safety._dir_size(tmp_path)
        assert result >= 300

    def test_handles_oserror_on_stat(self, tmp_path):
        (tmp_path / "ok.txt").write_bytes(b"z" * 50)
        with patch.object(Path, "rglob") as mock_rglob:
            bad_path = MagicMock()
            bad_path.stat.side_effect = OSError("permission denied")
            good_path = MagicMock()
            good_path.stat.return_value = MagicMock(st_size=50)
            mock_rglob.return_value = [bad_path, good_path]
            with patch.object(Path, "exists", return_value=True):
                result = compile_safety._dir_size(tmp_path)
        assert result == 50


class TestNeuronMemoryUsage:
    def test_returns_first_element_on_success(self):
        import json

        fake_data = [{"cores": 16, "hbm_used": 1024}]
        with patch("subprocess.check_output", return_value=json.dumps(fake_data).encode()):
            result = compile_safety._neuron_memory_usage()
        assert result == {"cores": 16, "hbm_used": 1024}

    def test_returns_none_on_empty_list(self):
        import json

        with patch("subprocess.check_output", return_value=json.dumps([]).encode()):
            result = compile_safety._neuron_memory_usage()
        assert result is None

    def test_returns_none_on_exception(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = compile_safety._neuron_memory_usage()
        assert result is None

    def test_returns_none_on_non_list(self):
        import json

        with patch("subprocess.check_output", return_value=json.dumps({"key": "val"}).encode()):
            result = compile_safety._neuron_memory_usage()
        assert result is None


class TestCleanupStaleNeuronxccDirs:
    def test_removes_old_directories(self, tmp_path):
        stale_dir = tmp_path / "neuronxcc-abc123"
        stale_dir.mkdir()
        (stale_dir / "file.bin").write_bytes(b"x" * 500)
        old_time = time.time() - 3 * 3600
        os.utime(stale_dir, (old_time, old_time))

        freed = compile_safety.cleanup_stale_neuronxcc_dirs(tmp_path, max_age_min=120.0)
        assert freed >= 500
        assert not stale_dir.exists()

    def test_keeps_recent_directories(self, tmp_path):
        recent_dir = tmp_path / "neuronxcc-fresh"
        recent_dir.mkdir()
        (recent_dir / "data.bin").write_bytes(b"y" * 200)

        freed = compile_safety.cleanup_stale_neuronxcc_dirs(tmp_path, max_age_min=120.0)
        assert freed == 0
        assert recent_dir.exists()

    def test_dry_run_does_not_remove(self, tmp_path):
        stale_dir = tmp_path / "neuronxcc-dryrun"
        stale_dir.mkdir()
        (stale_dir / "data.bin").write_bytes(b"z" * 100)
        old_time = time.time() - 3 * 3600
        os.utime(stale_dir, (old_time, old_time))

        freed = compile_safety.cleanup_stale_neuronxcc_dirs(
            tmp_path, max_age_min=120.0, dry_run=True
        )
        assert freed == 0
        assert stale_dir.exists()

    def test_handles_oserror_on_stat(self, tmp_path):
        bad_dir = tmp_path / "neuronxcc-badstat"
        bad_dir.mkdir()
        with patch.object(Path, "glob") as mock_glob:
            item = MagicMock()
            item.name = "neuronxcc-badstat"
            item.stat.side_effect = OSError("permission denied")
            mock_glob.return_value = [item]
            freed = compile_safety.cleanup_stale_neuronxcc_dirs(tmp_path, max_age_min=0)
        assert freed == 0

    def test_handles_rmtree_oserror(self, tmp_path):
        stale_dir = tmp_path / "neuronxcc-rmfail"
        stale_dir.mkdir()
        old_time = time.time() - 3 * 3600
        os.utime(stale_dir, (old_time, old_time))

        with patch("shutil.rmtree", side_effect=OSError("busy")):
            freed = compile_safety.cleanup_stale_neuronxcc_dirs(tmp_path, max_age_min=120.0)
        assert freed == 0


class TestDiskGuard:
    def test_creates_directory_on_enter(self, tmp_path):
        target = tmp_path / "guarded_dir"
        assert not target.exists()
        with compile_safety.DiskGuard(target, max_gb=10.0, interval_s=60.0):
            assert target.exists()

    def test_stops_thread_on_exit(self, tmp_path):
        target = tmp_path / "guarded_stop"
        with compile_safety.DiskGuard(target, max_gb=10.0, interval_s=60.0) as guard:
            assert guard._th is not None
            assert guard._th.is_alive()
        assert guard._stop.is_set()

    def test_aborts_when_size_exceeds_cap(self, tmp_path):
        target = tmp_path / "guarded_abort"
        target.mkdir()
        with patch.object(compile_safety, "_dir_size", return_value=int(50e9)):
            with patch("os._exit") as mock_exit:
                guard = compile_safety.DiskGuard(target, max_gb=10.0, interval_s=0.01)
                guard.__enter__()
                time.sleep(0.1)
                guard.__exit__(None, None, None)
                mock_exit.assert_called_with(137)

    def test_does_not_abort_when_under_cap(self, tmp_path):
        target = tmp_path / "guarded_ok"
        target.mkdir()
        with patch.object(compile_safety, "_dir_size", return_value=int(1e9)):
            with patch("os._exit") as mock_exit:
                with compile_safety.DiskGuard(target, max_gb=10.0, interval_s=0.01):
                    time.sleep(0.05)
                mock_exit.assert_not_called()


class TestResourceMonitor:
    def setup_method(self):
        compile_safety._MONITOR_THREAD = None
        compile_safety._MONITOR_STOP = threading.Event()

    def teardown_method(self):
        self._stop_monitor()
        compile_safety._MONITOR_THREAD = None
        compile_safety._MONITOR_STOP = threading.Event()

    @staticmethod
    def _stop_monitor():
        # Stop and join the monitor thread while the test's patches are still
        # active. If it outlives them, its next tick runs against the real
        # _bytes_free/_mem_available_bytes and the real os._exit, which kills
        # the whole pytest process on a host with < 20 GB free disk.
        compile_safety.stop_resource_monitor()
        if compile_safety._MONITOR_THREAD is not None:
            compile_safety._MONITOR_THREAD.join(timeout=2)
            assert not compile_safety._MONITOR_THREAD.is_alive()

    def test_starts_daemon_thread(self):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                compile_safety.install_resource_monitor(
                    limits=compile_safety.Limits(interval_s=0.01)
                )
                assert compile_safety._MONITOR_THREAD is not None
                assert compile_safety._MONITOR_THREAD.is_alive()
                assert compile_safety._MONITOR_THREAD.daemon is True
                self._stop_monitor()

    def test_noop_on_second_call(self):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                compile_safety.install_resource_monitor(
                    limits=compile_safety.Limits(interval_s=0.01)
                )
                first_thread = compile_safety._MONITOR_THREAD
                compile_safety.install_resource_monitor(
                    limits=compile_safety.Limits(interval_s=0.01)
                )
                assert compile_safety._MONITOR_THREAD is first_thread
                self._stop_monitor()

    def test_stop_sets_event(self):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                compile_safety.install_resource_monitor(
                    limits=compile_safety.Limits(interval_s=0.01)
                )
                compile_safety.stop_resource_monitor()
                assert compile_safety._MONITOR_STOP.is_set()
                self._stop_monitor()

    def test_aborts_on_low_disk(self):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(5e9)):
                with patch("os._exit") as mock_exit:
                    compile_safety.install_resource_monitor(
                        limits=compile_safety.Limits(interval_s=0.01, min_disk_gb=20.0)
                    )
                    time.sleep(0.1)
                    self._stop_monitor()
                    mock_exit.assert_called_with(137)

    def test_aborts_on_low_ram(self):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(2e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                with patch("os._exit") as mock_exit:
                    compile_safety.install_resource_monitor(
                        limits=compile_safety.Limits(interval_s=0.01, min_ram_gb=10.0)
                    )
                    time.sleep(0.1)
                    self._stop_monitor()
                    mock_exit.assert_called_with(137)

    def test_aborts_on_large_workdir(self, tmp_path):
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                with patch.object(compile_safety, "_dir_size", return_value=int(70e9)):
                    with patch("os._exit") as mock_exit:
                        compile_safety.install_resource_monitor(
                            limits=compile_safety.Limits(interval_s=0.01, max_workdir_gb=60.0),
                            watch_dirs=[str(tmp_path)],
                        )
                        time.sleep(0.1)
                        self._stop_monitor()
                        mock_exit.assert_called_with(137)

    def test_writes_to_log_file(self, tmp_path):
        log_file = tmp_path / "monitor.log"
        with patch.object(compile_safety, "_mem_available_bytes", return_value=int(32e9)):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                compile_safety.install_resource_monitor(
                    log_path=log_file,
                    limits=compile_safety.Limits(interval_s=0.01),
                )
                time.sleep(0.1)
                self._stop_monitor()
        assert log_file.exists()
        content = log_file.read_text()
        assert "[monitor]" in content

    def test_survives_tick_exception(self):
        call_count = {"n": 0}

        def flaky_mem():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient failure")
            return int(32e9)

        with patch.object(compile_safety, "_mem_available_bytes", side_effect=flaky_mem):
            with patch.object(compile_safety, "_bytes_free", return_value=int(100e9)):
                compile_safety.install_resource_monitor(
                    limits=compile_safety.Limits(interval_s=0.01)
                )
                time.sleep(0.1)
                assert compile_safety._MONITOR_THREAD.is_alive()
                self._stop_monitor()


class TestServingSafety:
    @pytest.fixture(autouse=True)
    def _import_serving(self):
        serving_path = str(Path(__file__).resolve().parents[4] / "runtime" / "serving")
        if serving_path not in sys.path:
            sys.path.insert(0, serving_path)
        import _safety as serving_safety

        self.mod = serving_safety
        yield
        self.mod._MONITOR_THREAD = None
        self.mod._MONITOR_STOP = threading.Event()

    def test_limits_defaults(self):
        lim = self.mod.Limits()
        assert lim.min_ram_gb == 10.0
        assert lim.min_disk_gb == 20.0

    def test_bytes_free_success(self):
        fake_stat = MagicMock()
        fake_stat.f_bavail = 500
        fake_stat.f_frsize = 4096
        with patch("os.statvfs", return_value=fake_stat):
            result = self.mod._bytes_free("/test")
        assert result == 500 * 4096

    def test_cleanup_stale_dirs(self, tmp_path):
        stale = tmp_path / "neuronxcc-old"
        stale.mkdir()
        (stale / "x.bin").write_bytes(b"a" * 100)
        old_time = time.time() - 4 * 3600
        os.utime(stale, (old_time, old_time))

        freed = self.mod.cleanup_stale_neuronxcc_dirs(tmp_path, max_age_min=60)
        assert freed >= 100
        assert not stale.exists()
