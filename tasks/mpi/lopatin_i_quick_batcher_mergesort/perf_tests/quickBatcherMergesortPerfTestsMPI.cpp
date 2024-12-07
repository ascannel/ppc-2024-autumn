#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/lopatin_i_quick_batcher_mergesort/include/quickBatcherMergesortHeaderMPI.hpp"

std::vector<int> generateArray(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> outputArray(size);
  for (int i = 0; i < size; i++) {
    outputArray[i] = (gen() % 200) - 99;
  }
  return outputArray;
}

std::vector<int> testData = generateArray(10000);

TEST(lopatin_i_quick_batcher_mergesort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> inputArray = testData;
  std::vector<int> resultArray(10000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
    taskDataParallel->inputs_count.emplace_back(inputArray.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
    taskDataParallel->outputs_count.emplace_back(resultArray.size());
  }

  auto testTask = std::make_shared<lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskParallel>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(lopatin_i_quick_batcher_mergesort_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> inputArray = testData;
  std::vector<int> resultArray(10000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
    taskDataParallel->inputs_count.emplace_back(inputArray.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
    taskDataParallel->outputs_count.emplace_back(resultArray.size());
  }

  auto testTask = std::make_shared<lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskParallel>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}