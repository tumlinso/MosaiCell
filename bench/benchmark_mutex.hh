#pragma once

#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellshard_preprocess::bench {

class benchmark_mutex_guard {
public:
    explicit benchmark_mutex_guard(const char *name = "gpu-bench")
        : benchmark_mutex_guard(name, nullptr, 0u) {}

    benchmark_mutex_guard(const char *name, int device_id)
        : benchmark_mutex_guard(name, &device_id, 1u) {}

    benchmark_mutex_guard(const char *name, const int *device_ids, std::size_t device_count) {
        const char *tag = name != nullptr ? name : "gpu-bench";
        const char *override_path = std::getenv("CUDA_V100_BENCHMARK_MUTEX_PATH");
        if (override_path != nullptr && *override_path != '\0') {
            acquire_path_(override_path, tag);
            return;
        }
        if (device_ids == nullptr || device_count == 0u) {
            acquire_path_("/tmp/cuda_v100_benchmark.lock", tag);
            return;
        }
        std::vector<int> sorted(device_ids, device_ids + device_count);
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        for (int device : sorted) {
            char path[256];
            std::snprintf(path, sizeof(path), "/tmp/cuda_v100_benchmark.device%d.lock", device);
            acquire_path_(path, tag);
        }
    }

    ~benchmark_mutex_guard() {
        for (std::size_t i = 0u; i < fds_.size(); ++i) {
            if (fds_[i] >= 0) {
                ::flock(fds_[i], LOCK_UN);
                ::close(fds_[i]);
            }
        }
    }

    benchmark_mutex_guard(const benchmark_mutex_guard &) = delete;
    benchmark_mutex_guard &operator=(const benchmark_mutex_guard &) = delete;

private:
    void acquire_path_(const char *path, const char *tag) {
        const int fd = ::open(path, O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            throw std::runtime_error(std::string("failed to open benchmark mutex: ") + std::strerror(errno));
        }
        if (::flock(fd, LOCK_EX) != 0) {
            const int saved_errno = errno;
            ::close(fd);
            throw std::runtime_error(std::string("failed to lock benchmark mutex: ") + std::strerror(saved_errno));
        }
        fds_.push_back(fd);
        std::fprintf(stderr, "[benchmark-mutex] acquired %s via %s\n", tag, path);
    }

    std::vector<int> fds_;
};

} // namespace cellshard_preprocess::bench
