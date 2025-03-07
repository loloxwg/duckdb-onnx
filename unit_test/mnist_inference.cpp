#include <duckdb.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat preprocess_image(const std::string &image_path) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Error: Unable to load image!" << std::endl;
    exit(1);
  }
  cv::resize(image, image, cv::Size(28, 28));
  image.convertTo(image, CV_32F, 1.0 / 255.0);
  return image;
}

int main() {
  cv::Mat image = preprocess_image("mnist/images/7_12.png");
  if (!image.isContinuous()) {
    image = image.clone();
  }
  std::vector<float> input_data(image.ptr<float>(),
                                image.ptr<float>() + image.total());

  duckdb_database db;
  duckdb_connection con;

  if (duckdb_open(NULL, &db) == DuckDBError) {
    // handle error
  }
  if (duckdb_connect(db, &con) == DuckDBError) {
    // handle error
  }

  // run queries...
  duckdb_result res;

  std::string tensor_value = "[";
  for (int i = 0; i < input_data.size(); ++i) {
    tensor_value += std::to_string(input_data[i]);
    if (i != input_data.size() - 1) {
      tensor_value += ",";
    }
  }

  std::string sql = "SELECT "
                    "onnx('mnist/onnx/mnist-8.onnx',"
                    "{'shape':[1,1,28,28],'value':" +
                    tensor_value + "]}) as result";

  duckdb_state state = duckdb_query(con, sql.c_str(), &res);
  if (state == DuckDBError) {
    // handle error
    std::cerr << "Error" << std::endl;
    exit(1);
  }

  std::vector<float> output_data;
  output_data.reserve(10);
  while (true) {
    duckdb_data_chunk result = duckdb_fetch_chunk(res);
    if (!result) {
      // result is exhausted
      break;
    }
    // get the number of rows from the data chunk
    idx_t row_count = duckdb_data_chunk_get_size(result);
    assert(row_count == 1);
    // get the first column
    duckdb_vector struct_col = duckdb_data_chunk_get_vector(result, 0);
    uint64_t *struct_validity = duckdb_vector_get_validity(struct_col);

    duckdb_vector col1_vector = duckdb_struct_vector_get_child(struct_col, 0);
    duckdb_vector col2_vector = duckdb_struct_vector_get_child(struct_col, 1);

    duckdb_list_entry *list_data_1 =
        (duckdb_list_entry *)duckdb_vector_get_data(col1_vector);
    uint64_t *list_validity_1 = duckdb_vector_get_validity(col1_vector);
    // get the child column of the list
    duckdb_vector list_child_1 = duckdb_list_vector_get_child(col1_vector);
    int32_t *child_data_1 = (int32_t *)duckdb_vector_get_data(list_child_1);
    uint64_t *child_validity_1 = duckdb_vector_get_validity(list_child_1);

    duckdb_list_entry *list_data_2 =
        (duckdb_list_entry *)duckdb_vector_get_data(col2_vector);
    uint64_t *list_validity_2 = duckdb_vector_get_validity(col2_vector);
    // get the child column of the list
    duckdb_vector list_child_2 = duckdb_list_vector_get_child(col2_vector);
    float *child_data_2 = (float *)duckdb_vector_get_data(list_child_2);
    uint64_t *child_validity_2 = duckdb_vector_get_validity(list_child_2);

    for (idx_t row = 0; row < row_count; row++) {
      if (!duckdb_validity_row_is_valid(list_validity_1, row)) {
        // entire list is NULL
        printf("NULL\n");
        continue;
      }
      // read the list offsets for this row
      duckdb_list_entry list = list_data_1[row];
      printf("[");
      for (idx_t child_idx = list.offset; child_idx < list.offset + list.length;
           child_idx++) {
        if (child_idx > list.offset) {
          printf(", ");
        }
        if (!duckdb_validity_row_is_valid(child_validity_1, child_idx)) {
          // col1 is NULL
          printf("NULL");
        } else {
          printf("%lld", child_data_1[child_idx]);
        }
      }
      printf("]\n");

      duckdb_list_entry list2 = list_data_2[row];
      printf("[");
      for (idx_t child_idx = list2.offset;
           child_idx < list2.offset + list2.length; child_idx++) {
        if (child_idx > list2.offset) {
          printf(", ");
        }
        if (!duckdb_validity_row_is_valid(child_validity_2, child_idx)) {
          // col2 is NULL
          printf("NULL");
        } else {
          printf("%f", child_data_2[child_idx]);
          output_data.push_back(child_data_2[child_idx]);
        }
      }
      printf("]\n");
    }
    duckdb_destroy_data_chunk(&result);
  }

  // clean-up
  duckdb_destroy_result(&res);
  duckdb_disconnect(&con);
  duckdb_close(&db);

  std::cout << "Output probabilities: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  //
  int predicted_class =
      std::distance(output_data.begin(),
                    std::max_element(output_data.begin(), output_data.end()));
  std::cout << "Predicted Class: " << predicted_class << std::endl;

  return 0;
}
