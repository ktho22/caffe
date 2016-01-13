#include <cfloat>
#include <vector>

#include "caffe/layers/scalar_mult_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::Forward_gpu(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
            const int count = top[0]->count();
            Dtype* top_data = top[0]->mutable_gpu_data();
            Dtype* Scale_data = bottom[0]->gpu_data()

                for(int i = 0; i < count; i ++){
                    caffe_gpu_scale(count, Scale_data[i], bottom[1]->gpu_data()[i], top_data[i]);
                }
        }

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            const int count = top[0]->count();
            const Dtype* top_data = top[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();

            // Bottom 0 is Scalar weight
            if (propagate_down[0]){
                const Dtype* bottom_data = bottom[1]->gpu_data(); // Vector data
                Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
                caffe_copy(count, bottom_data, bottom_diff);
                for(int i = 0; i < count; i++){
                    bottom_diff[i] = caffe_gpu_dot(count, bottom_diff[i], top_diff[i]);
                }
            }

            // Bottom 1 is Vector data
            if (propagate_down[1]){
                const Dtype* bottom_data = bottom[0]->gpu_data(); // Scalar weight
                Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
                for(int i = 0; i < count; i++){
                    caffe_gpu_scale(count, bottom_data[i], top_diff[i], bottom_diff[i]);
                }
            }
        }

    INSTANTIATE_LAYER_GPU_FUNCS(ScalarMultLayer);
}  // namespace caffe
