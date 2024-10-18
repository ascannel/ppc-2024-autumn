#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

int countWords(const std::string& str) {
  std::istringstream iss(str);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

std::string generateLongString(int n) {
  std::string testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    testData += testString;
  }
  return testData;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  wordCount = 0;
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  wordCount = countWords(input_);
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  wordCount = 0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  int inputLength = input_.length();
  boost::mpi::broadcast(world, inputLength, 0);

  if (world.rank() != 0) {
    input_.resize(inputLength);
  }
  boost::mpi::broadcast(world, input_, 0);

  int totalWords = 0;
  std::vector<std::string> words;

  std::istringstream iss(input_);
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }

  totalWords = words.size();

  // debug
  // std::cout << "Process " << world.rank() << ": total_words = " << totalWords << std::endl;

  int localWordsCount = totalWords / world.size();
  int remainder = totalWords % world.size();

  int start = world.rank() * localWordsCount + std::min(world.rank(), remainder);
  int end = start + localWordsCount + (world.rank() < remainder ? 1 : 0);

  // debug
  // std::cout << "Process " << world.rank() << ": start = " << start << ", end = " << end << std::endl;

  // Проверяем, что start и end находятся в пределах words
  //if (start < 0 || start >= words.size()) {
  //  std::cerr << "Error: start is out of bounds!" << std::endl;
  //  return false;
  //}
  //if (end < 0 || end > words.size()) {
  //  std::cerr << "Error: end is out of bounds!" << std::endl;
  //  return false;
  //}

  int localWordCount = end - start;
  if (start < totalWords) {
    localWordCount =
        std::count_if(words.begin() + start, words.begin() + end, [](const std::string& w) { return !w.empty(); });
  } else {
    localWordCount = 0;
  }

  // debug
  // std::cout << "Process " << world.rank() << ": local_word_count = " << localWordCount << std::endl;

  boost::mpi::reduce(world, localWordCount, wordCount, std::plus<int>(), 0);

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  }
  return true;
}

}  // namespace lopatin_i_count_words_mpi