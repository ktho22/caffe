#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/scalar_mult_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScalarMultTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ScalarMultTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ScalarMultTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScalarMultTest, TestDtypesAndDevices);

TYPED_TEST(ScalarMultTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<ScalarMultLayer<Dtype> > layer(
      new ScalarMultLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(ScalarMultTest, TestProd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<ScalarMultLayer<Dtype> > layer(
      new ScalarMultLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] * in_data_b[i] * in_data_c[i], 1e-4);
  }
}

TYPED_TEST(ScalarMultTest, TestUnstableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ScalarMultLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
