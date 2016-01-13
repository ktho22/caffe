#include <cfloat>
#include <vector>

#include "caffe/layers/scalar_mult_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
        }

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
            top[0]->ReshapeLike(*bottom[1]);
        }

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::Forward_cpu(
                const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
            const int num = bottom[1]->num();
            const int dim = bottom[1]->count() / num;
            Dtype* top_data = top[0]->mutable_cpu_data();
            const Dtype* Scale_data = bottom[0]->cpu_data();

            for (int i = 0; i < num; ++i) {
                caffe_cpu_scale(dim, Scale_data[i],
                        bottom[1]->mutable_cpu_data() + bottom[1]->offset(i),
                        top_data + top[0]->offset(i));
            }
        }

    template <typename Dtype>
        void ScalarMultLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            const int count = top[0]->count();
            const int num = bottom[1]->num();
            const int dim = bottom[1]->count() / num;
            const Dtype* top_diff = top[0]->cpu_diff();

            // Bottom 0 is Scalar weight
            if (propagate_down[0]){
                const Dtype* bottom_data = bottom[1]->cpu_data(); // Vector data
                Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
                caffe_copy(count, bottom_data, bottom_diff);
                for(int i = 0; i < num; i++){
                    bottom_diff[i] = caffe_cpu_dot(dim, bottom_diff+bottom[0]->offset(i), 
                            top_diff + top[0]->offset(i));
                }
            }

            // Bottom 1 is Vector data
            if (propagate_down[1]){
                const Dtype* bottom_data = bottom[0]->cpu_data(); // Scalar weight
                Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
                for(int i = 0; i < num; i++){
                    caffe_cpu_scale(dim, bottom_data[i], top_diff + top[0]->offset(i), 
                            bottom_diff + bottom[1]->offset(i));
                }
            }
        }


#ifdef CPU_ONLY
            STUB_GPU(ScalarMultLayer);
#endif

            INSTANTIATE_CLASS(ScalarMultLayer);
            REGISTER_LAYER_CLASS(ScalarMult);

        }  // namespace caffe
