#include <gtest/gtest.h>

#include "seq/lopatin_i_quick_batcher_mergesort/include/quickBatcherMergesortHeaderSeq.hpp"

std::vector<int> generateArray(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> outputArray(size);
  for (int i = 0; i < size; i++) {
    outputArray[i] = (gen() % 200) - 99;
  }
  return outputArray;
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_validation_empty_array) {
  std::vector<int> inputArray = {};
  std::vector<int> resultArray(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_FALSE(testTask.validation());
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_validation_1_int) {
  std::vector<int> inputArray = {1};
  std::vector<int> resultArray(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_FALSE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_2_int) {
  std::vector<int> inputArray = {4, 3};
  std::vector<int> resultArray(2, 0);
  std::vector<int> expectedResult = {3, 4};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_3_int) {
  std::vector<int> inputArray = {2, 1, 3};
  std::vector<int> resultArray(3, 0);
  std::vector<int> expectedResult = {1, 2, 3};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_10_int) {
  std::vector<int> inputArray = generateArray(10);
  std::vector<int> resultArray(10, 0);
  std::vector<int> expectedResult = inputArray;
  std::sort(expectedResult.begin(), expectedResult.end());

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_reverse_10_int) {
  std::vector<int> inputArray = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> resultArray(10, 0);
  std::vector<int> expectedResult = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_sorted_10_int) {
  std::vector<int> inputArray = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> resultArray(10, 0);
  std::vector<int> expectedResult = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_100_int) {
  std::vector<int> inputArray = generateArray(100);
  std::vector<int> resultArray(100, 0);
  std::vector<int> expectedResult = inputArray;
  std::sort(expectedResult.begin(), expectedResult.end());

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_1000_int) {
  std::vector<int> inputArray = generateArray(1000);
  std::vector<int> resultArray(1000, 0);
  std::vector<int> expectedResult = inputArray;
  std::sort(expectedResult.begin(), expectedResult.end());

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  EXPECT_EQ(resultArray, expectedResult);
}