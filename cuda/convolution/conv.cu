#include <cudnn.h>
#include <opencv2/opencv.hpp>

/* A macro that checks the status object for any
error condition and aborts the execution of the
program if something went wrong. We can then
simply wrap any library function we call with
that macro.
*/
#define checkCUDNN(expression) \
{ \
cudnnStatus_t = status = (expression);\
if (status != CUDNN_STATUS_SUCCESS) { \
    std::cerr << "Error on line " << __LINE__ \
    << ": " \
    << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE); \
} \
} \

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path,
    CV_LOAG_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

int main(int argc, char const *argv[]) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cv::Mat image = load_image(argv[1]);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
    /*format=*/))
}

