#include <gtest/gtest.h>

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

TEST(lopatin_i_quick_batcher_mergesort_mpi, test_validation_empty_array) {
  boost::mpi::communicator world;
  std::vector<int> inputArray = {};
  std::vector<int> resultArray(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
    taskDataParallel->inputs_count.emplace_back(inputArray.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
    taskDataParallel->outputs_count.emplace_back(resultArray.size());

    lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_quick_batcher_mergesort_mpi, test_validation_1_int) {
  boost::mpi::communicator world;
  std::vector<int> inputArray = {1};
  std::vector<int> resultArray(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
    taskDataParallel->inputs_count.emplace_back(inputArray.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
    taskDataParallel->outputs_count.emplace_back(resultArray.size());

    lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

//TEST(lopatin_i_quick_batcher_mergesort_mpi, test_10_int) {
//  boost::mpi::communicator world;
//  std::vector<int> inputArray = generateArray(10);
//  std::vector<int> resultArray(10, 0);
//
//  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
//    taskDataParallel->inputs_count.emplace_back(inputArray.size());
//    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
//    taskDataParallel->outputs_count.emplace_back(resultArray.size());
//  }
//
//  lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
//  ASSERT_TRUE(testTaskParallel.validation());
//  testTaskParallel.pre_processing();
//  testTaskParallel.run();
//  testTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    std::vector<int> referenceArray(10, 0);
//    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
//
//    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
//    taskDataSequential->inputs_count.emplace_back(inputArray.size());
//    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceArray.data()));
//    taskDataSequential->outputs_count.emplace_back(referenceArray.size());
//
//    lopatin_i_quick_batcher_mergesort_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
//    ASSERT_TRUE(testTaskSequential.validation());
//    testTaskSequential.pre_processing();
//    testTaskSequential.run();
//    testTaskSequential.post_processing();
//
//    EXPECT_EQ(resultArray, referenceArray);
//  }
//}