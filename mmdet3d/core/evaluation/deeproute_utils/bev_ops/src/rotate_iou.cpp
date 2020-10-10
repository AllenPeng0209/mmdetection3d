#include <torch/extension.h>
#include <boost/geometry.hpp>
#include <vector>
#include <iostream>
#include <time.h>

using namespace std;

struct Point {
  Point(float a, float b): x(a), y(b) {};
  float x, y;
  float cross(const Point& p1) {
    return x * p1.y - y * p1.x;
  }
};

struct Line {
  Line(Point p1, Point p2) {
    a = p2.y - p1.y;
    b = p1.x - p2.x;
    c = p2.cross(p1);
  };
 public:
  Point InterPt(const Line& l2) {
    float w = a * l2.b - b * l2.a;
    float x1 = (b * l2.c - c * l2.b)/w;
    float y1 = (c * l2.a - a * l2.c)/w;
    return Point(x1, y1);
  }
  float LineValue(const Point& p) {
    return a * p.x + b * p.y + c;
  }

 private:
  float a, b, c;


};

float GetIntersection(const std::vector<Point>& rect1, const std::vector<Point>& rect2) {

  std::vector<Point> intersection = rect1;
  int cnt = 0;
  for (int i = 0; i < 4; i++) {
    if (intersection.size() < 2)
      break;
    Line line = Line(rect2.at(i), rect2.at((i + 1) % rect2.size()));
    std::vector<float> line_v;
    for (int j = 0; j < intersection.size(); ++j) {
      line_v.emplace_back(line.LineValue(intersection.at(j)));
      //std::cout << cnt << " " << line_v[-1] << std::endl;
    }

    std::vector<Point> new_intersec;
    for (int j = 0; j < line_v.size(); ++j) {
      cnt += 1;
      float s_value = line_v[j];
      float t_value = line_v[(j + 1) % (line_v.size())];
      if (s_value <= 0)
        new_intersec.push_back(intersection[j]);
      if (s_value * t_value < 0) {
        Point inter_pt = line.InterPt(Line(intersection[j],
            intersection[(j + 1) % intersection.size()]));
        new_intersec.push_back(inter_pt);
      }
    }
    intersection = new_intersec;
  }
  if (intersection.size() < 2)
    return 0;
  float iou = 0.;
  for (int i = 0; i < intersection.size(); ++i) {
    Point p1 = intersection[i];
    Point p2 = intersection[(i + 1) % intersection.size()];
    iou += (p1.x * p2.y - p1.y * p2.x);
  }

  return iou * 0.5;
}

template <typename scalar_t>
at::Tensor rotated_iou_kernel_v2(const at::Tensor& input1, const at::Tensor& input2) {
  AT_ASSERTM(!input1.type().is_cuda(), "dets must be a CPU tensor");
  if (input1.numel() == 0 || input2.numel() == 0) {
    return at::empty({0, 0}, input1.options().dtype(at::kFloat).device(at::kCPU));
  }

  auto n = input1.size(0);
  auto m = input2.size(0);
  at::Tensor output_iou =
      at::zeros({n, m}, input1.options().dtype(at::kFloat).device(at::kCPU));
  auto input1_x = input1.select(2, 0);
  auto input1_y = input1.select(2, 1);

  auto input2_x = input2.select(2, 0);
  auto input2_y = input2.select(2, 1);
  std::vector<at::Tensor> input1_x_tensor;
  std::vector<at::Tensor> input1_y_tensor;
  std::vector<at::Tensor> input2_x_tensor;
  std::vector<at::Tensor> input2_y_tensor;

  for (size_t i = 0 ; i < 4; ++i) {
    input1_x_tensor.emplace_back(input1_x.select(1, i).contiguous());
    input1_y_tensor.emplace_back(input1_y.select(1, i).contiguous());
    input2_x_tensor.emplace_back(input2_x.select(1, i).contiguous());
    input2_y_tensor.emplace_back(input2_y.select(1, i).contiguous());
  }

  std::vector<Point> rect1;
  std::vector<Point> rect2;
  for (int i = 0; i < n; ++i) {
    auto x1_min = input1_x_tensor[0].data<scalar_t>()[i];
    auto x1_max = input1_x_tensor[0].data<scalar_t>()[i];
    auto y1_min = input1_y_tensor[0].data<scalar_t>()[i];
    auto y1_max = input1_y_tensor[0].data<scalar_t>()[i];

    for (int k = 0; k < 4; k++) {
      x1_min = std::min(x1_min, input1_x_tensor[k].data<scalar_t>()[i]);
      x1_max = std::max(x1_max, input1_x_tensor[k].data<scalar_t>()[i]);
      y1_min = std::min(y1_min, input1_y_tensor[k].data<scalar_t>()[i]);
      y1_max = std::max(y1_max, input1_y_tensor[k].data<scalar_t>()[i]);
      rect1.emplace_back(Point(input1_x_tensor[k].data<scalar_t>()[i], input1_y_tensor[k].data<scalar_t>()[i]));
    }

    if (x1_max <= x1_min || y1_max <= y1_min)
      continue;
    float area1 = GetIntersection(rect1, rect1);
    //std::cout << "area1 " << area1 << std::endl;
    for (int j = 0; j < m; ++j) {

      auto x2_min = input2_x_tensor[0].data<scalar_t>()[j];
      auto x2_max = input2_x_tensor[0].data<scalar_t>()[j];
      auto y2_min = input2_y_tensor[0].data<scalar_t>()[j];
      auto y2_max = input2_y_tensor[0].data<scalar_t>()[j];

      for (int k = 0; k < 4; k++) {
        x2_min = std::min(x2_min, input2_x_tensor[k].data<scalar_t>()[j]);
        x2_max = std::max(x2_max, input2_x_tensor[k].data<scalar_t>()[j]);
        y2_min = std::min(y2_min, input2_y_tensor[k].data<scalar_t>()[j]);
        y2_max = std::max(y2_max, input2_y_tensor[k].data<scalar_t>()[j]);
      }
      if (x2_max <= x2_min || y2_max <= y2_min)
        continue;
      auto xmin = std::max(x1_min, x2_min);
      auto xmax = std::min(x1_max, x2_max);
      auto ymin = std::max(y1_min, y2_min);
      auto ymax = std::min(y1_max, y2_max);

      if ((ymax - ymin) * (xmax - xmin) <= 0)
        continue;
      for (int k = 0; k < 4; k++) {
        rect2.emplace_back(Point(input2_x_tensor[k].data<scalar_t>()[j], input2_y_tensor[k].data<scalar_t>()[j]));
      }
      float area2 = GetIntersection(rect2, rect2);
      float inter = GetIntersection(rect1, rect2);
      if (area2 == 0)
        continue;
      if (inter == 0)
        continue;

      output_iou[i][j] = inter / (area1 + area2 - inter);
      std::vector<Point>().swap(rect2);
    }
    std::vector<Point>().swap(rect1);
  }
  return output_iou;

}


template <typename scalar_t>
at::Tensor rotated_iou_kernel(const at::Tensor& input1, const at::Tensor& input2) {
  AT_ASSERTM(!input1.type().is_cuda(), "dets must be a CPU tensor");
  if (input1.numel() == 0 || input2.numel() == 0) {
    return at::empty({0, 0}, input1.options().dtype(at::kFloat).device(at::kCPU));
  }

  auto n = input1.size(0);
  auto m = input2.size(0);
  at::Tensor output_iou =
      at::zeros({n, m}, input1.options().dtype(at::kFloat).device(at::kCPU));
  auto input1_x = input1.select(2, 0);
  auto input1_y = input1.select(2, 1);

  auto input2_x = input2.select(2, 0);
  auto input2_y = input2.select(2, 1);
  std::vector<at::Tensor> input1_x_tensor;
  std::vector<at::Tensor> input1_y_tensor;
  std::vector<at::Tensor> input2_x_tensor;
  std::vector<at::Tensor> input2_y_tensor;

  for (size_t i = 0 ; i < 4; ++i) {
    input1_x_tensor.emplace_back(input1_x.select(1, i).contiguous());
    input1_y_tensor.emplace_back(input1_y.select(1, i).contiguous());
    input2_x_tensor.emplace_back(input2_x.select(1, i).contiguous());
    input2_y_tensor.emplace_back(input2_y.select(1, i).contiguous());
  }

  namespace bg = boost::geometry;
  typedef bg::model::point<scalar_t, 2, bg::cs::cartesian> point_t;
  typedef bg::model::ring<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  scalar_t inter_area, union_area;
  for (int i = 0; i < n; ++i) {
    auto x1_min = input1_x_tensor[0].data<scalar_t>()[i];
    auto x1_max = input1_x_tensor[0].data<scalar_t>()[i];
    auto y1_min = input1_y_tensor[0].data<scalar_t>()[i];
    auto y1_max = input1_y_tensor[0].data<scalar_t>()[i];

    for (int k = 0; k < 4; k++) {
      x1_min = std::min(x1_min, input1_x_tensor[k].data<scalar_t>()[i]);
      x1_max = std::max(x1_max, input1_x_tensor[k].data<scalar_t>()[i]);
      y1_min = std::min(y1_min, input1_y_tensor[k].data<scalar_t>()[i]);
      y1_max = std::max(y1_max, input1_y_tensor[k].data<scalar_t>()[i]);
      bg::append(poly, point_t(input1_x_tensor[k].data<scalar_t>()[i], input1_y_tensor[k].data<scalar_t>()[i]));
    }

    if (x1_max <= x1_min || y1_max <= y1_min)
      continue;
    bg::append(poly, point_t(input1_x_tensor[0].data<scalar_t>()[i], input1_y_tensor[0].data<scalar_t>()[i]));
    bg::correct(poly);
    for (int j = 0; j < m; ++j) {

      auto x2_min = input2_x_tensor[0].data<scalar_t>()[j];
      auto x2_max = input2_x_tensor[0].data<scalar_t>()[j];
      auto y2_min = input2_y_tensor[0].data<scalar_t>()[j];
      auto y2_max = input2_y_tensor[0].data<scalar_t>()[j];

      for (int k = 0; k < 4; k++) {
        x2_min = std::min(x2_min, input2_x_tensor[k].data<scalar_t>()[j]);
        x2_max = std::max(x2_max, input2_x_tensor[k].data<scalar_t>()[j]);
        y2_min = std::min(y2_min, input2_y_tensor[k].data<scalar_t>()[j]);
        y2_max = std::max(y2_max, input2_y_tensor[k].data<scalar_t>()[j]);
      }

      auto xmin = std::max(x1_min, x2_min);
      auto xmax = std::min(x1_max, x2_max);
      auto ymin = std::max(y1_min, y2_min);
      auto ymax = std::min(y1_max, y2_max);


      if ((ymax - ymin) * (xmax - xmin) <= 0)
        continue;
      for (int k = 0; k < 4; k++) {

        bg::append(qpoly, point_t(input2_x_tensor[k].data<scalar_t>()[j], input2_y_tensor[k].data<scalar_t>()[j]));
      }

      bg::append(qpoly, point_t(input2_x_tensor[0].data<scalar_t>()[j], input2_y_tensor[0].data<scalar_t>()[j]));
      bg::correct(qpoly);
      bg::intersection(poly, qpoly, poly_inter);

      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        // bg::union_(poly, qpoly, poly_union);
        // if (!poly_union.empty()) {
          float area1 = bg::area(qpoly);
          float area2 =  bg::area(poly);
          // std::cout << "area 1 " << area1 << " " << area2 << " "<< inter_area <<  std::endl;
          output_iou[i][j] = inter_area / (area1 + area2 - inter_area);
        // }
        // poly_union.clear();
      }
      // poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
    poly.clear();
  }
  return output_iou;

}

template <typename scalar_t>
at::Tensor rotated_iou_kernel_v3(const at::Tensor& input1, const at::Tensor& input2) {
  AT_ASSERTM(!input1.type().is_cuda(), "dets must be a CPU tensor");
  if (input1.numel() == 0 || input2.numel() == 0) {
    return at::empty({0, 0}, input1.options().dtype(at::kFloat).device(at::kCPU));
  }

  auto n = input1.size(0);
  auto m = input2.size(0);
  at::Tensor output_iou =
      at::zeros({n, m}, input1.options().dtype(at::kFloat).device(at::kCPU));
//  auto input1_x = input1.select(2, 0);
//  auto input1_y = input1.select(2, 1);
//
//  auto input2_x = input2.select(2, 0);
//  auto input2_y = input2.select(2, 1);
//  std::vector<at::Tensor> input1_x_tensor;
//  std::vector<at::Tensor> input1_y_tensor;
//  std::vector<at::Tensor> input2_x_tensor;
//  std::vector<at::Tensor> input2_y_tensor;
//
//  for (size_t i = 0 ; i < 4; ++i) {
//    input1_x_tensor.emplace_back(input1_x.select(1, i).contiguous());
//    input1_y_tensor.emplace_back(input1_y.select(1, i).contiguous());
//    input2_x_tensor.emplace_back(input2_x.select(1, i).contiguous());
//    input2_y_tensor.emplace_back(input2_y.select(1, i).contiguous());
//  }

  namespace bg = boost::geometry;
  typedef bg::model::point<scalar_t, 2, bg::cs::cartesian> point_t;
  typedef bg::model::ring<point_t> polygon_t;
//  std::cout << "number " << input1.numel() << " " << input2.numel() << std::endl;

  std::vector<scalar_t> v1(input1.data<scalar_t>(), input1.data<scalar_t>() + input1.numel());
  std::vector<scalar_t> v2(input2.data<scalar_t>(), input2.data<scalar_t>() + input2.numel());

//  for (size_t i = 0; i < v1.size(); ++i) {
//    std::cout << v1[i] << std::endl;
//  }
//  std::cout << " ------------ " << std::endl;
//  for (const auto& v: v1) {
//    std::cout << v << " " << std::endl;
//  }
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  scalar_t inter_area, union_area;
  for (int i = 0; i < n; ++i) {
    auto x1_min = v1[i * 4 * 2 + 2 * 0 + 0];
    auto x1_max = v1[i * 4 * 2 + 2 * 0 + 0];
    auto y1_min = v1[i * 4 * 2 + 2 * 0 + 1];
    auto y1_max = v1[i * 4 * 2 + 2 * 0 + 1];

    //std::cout << "x1 min " << x1_min << " " << x1_max << " " << y1_min << " " << y1_max << std::endl;

    for (int k = 0; k < 4; k++) {
      x1_min = std::min(x1_min, v1[i * 4 * 2 + 2 * k + 0]);
      x1_max = std::max(x1_max, v1[i * 4 * 2 + 2 * k + 0]);
      y1_min = std::min(y1_min, v1[i * 4 * 2 + 2 * k + 1]);
      y1_max = std::max(y1_max, v1[i * 4 * 2 + 2 * k + 1]);
      bg::append(poly, point_t(v1[i * 4 * 2 + 2 * k + 0], v1[i * 4 * 2 + 2 * k + 1]));
      //std::cout << "appending " << v1[i * 4 * 2 + 2 * k + 0] << " " << v1[i * 4 * 2 + 2 * k + 1] << std::endl;
    }

    if (x1_max <= x1_min || y1_max <= y1_min)
      continue;
    bg::append(poly, point_t(v1[i * 4 * 2 + 4 * 0 + 0], v1[i * 4 * 2 + 4 * 0 + 1]));
    bg::correct(poly);
    for (int j = 0; j < m; ++j) {

      auto x2_min = v2[j * 4 * 2 + 4 * 0 + 0];
      auto x2_max = v2[j * 4 * 2 + 4 * 0 + 0];;
      auto y2_min = v2[j * 4 * 2 + 4 * 0 + 1];;
      auto y2_max = v2[j * 4 * 2 + 4 * 0 + 1];;

      for (int k = 0; k < 4; k++) {
        x2_min = std::min(x2_min, v2[j * 4 * 2 + 2 * k + 0]);
        x2_max = std::max(x2_max, v2[j * 4 * 2 + 2 * k + 0]);
        y2_min = std::min(y2_min, v2[j * 4 * 2 + 2 * k + 1]);
        y2_max = std::max(y2_max, v2[j * 4 * 2 + 2 * k + 1]);
      }

      auto xmin = std::max(x1_min, x2_min);
      auto xmax = std::min(x1_max, x2_max);
      auto ymin = std::max(y1_min, y2_min);
      auto ymax = std::min(y1_max, y2_max);


      if ((ymax - ymin) * (xmax - xmin) <= 0)
        continue;
      for (int k = 0; k < 4; k++) {
        bg::append(qpoly, point_t(v2[j * 4 * 2 + 2 * k + 0], v2[j * 4 * 2 + 2 * k + 1]));
        //std::cout << "appending " << v2[j * 4 * 2 + 2 * k + 0] << " " << v2[j * 4 * 2 + 2 * k + 1] << std::endl;
      }

      bg::append(qpoly, point_t(v2[j * 4 * 2 + 4 * 0 + 0], v2[j * 4 * 2 + 4 * 0 + 1]));
      bg::correct(qpoly);
      bg::intersection(poly, qpoly, poly_inter);

      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        // bg::union_(poly, qpoly, poly_union);
        // if (!poly_union.empty()) {
          float area1 = bg::area(qpoly);
          float area2 =  bg::area(poly);
          // std::cout << "area 1 " << area1 << " " << area2 << " "<< inter_area <<  std::endl;
          output_iou[i][j] = inter_area / (area1 + area2 - inter_area);
        // }
        // poly_union.clear();
      }
      // poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
    poly.clear();
  }
  return output_iou;

}


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets, const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto scores = dets.select(1, 4).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold) suppressed[j] = 1;
    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

template <typename scalar_t>
at::Tensor rotated_iou_kernel_v4(const at::Tensor& input1, const at::Tensor& input2) {
  AT_ASSERTM(!input1.type().is_cuda(), "dets must be a CPU tensor");
  if (input1.numel() == 0 || input2.numel() == 0) {
    return at::empty({0, 0}, input1.options().dtype(at::kFloat).device(at::kCPU));
  }

  auto n = input1.size(0);
  auto m = input2.size(0);
  at::Tensor output_iou =
      at::zeros({n, m}, input1.options().dtype(at::kFloat).device(at::kCPU));

  //clock_t t1 = clock();
  std::vector<scalar_t> v1(input1.data<scalar_t>(), input1.data<scalar_t>() + input1.numel());
  std::vector<scalar_t> v2(input2.data<scalar_t>(), input2.data<scalar_t>() + input2.numel());
  //cout << "copy time " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
  std::vector<Point> rect1;
  std::vector<Point> rect2;
  for (int i = 0; i < n; ++i) {
    //t1 = clock();
    auto x1_min = v1[i * 4 * 2 + 2 * 0 + 0];
    auto x1_max = v1[i * 4 * 2 + 2 * 0 + 0];
    auto y1_min = v1[i * 4 * 2 + 2 * 0 + 1];
    auto y1_max = v1[i * 4 * 2 + 2 * 0 + 1];
    rect1.emplace_back(Point(v1[i * 4 * 2 + 2 * 0 + 0], v1[i * 4 * 2 + 2 * 0 + 1]));
    //std::cout << "x1 min " << x1_min << " " << x1_max << " " << y1_min << " " << y1_max << std::endl;

    for (int k = 1; k < 4; k++) {
      x1_min = std::min(x1_min, v1[i * 4 * 2 + 2 * k + 0]);
      x1_max = std::max(x1_max, v1[i * 4 * 2 + 2 * k + 0]);
      y1_min = std::min(y1_min, v1[i * 4 * 2 + 2 * k + 1]);
      y1_max = std::max(y1_max, v1[i * 4 * 2 + 2 * k + 1]);
      rect1.emplace_back(Point(v1[i * 4 * 2 + 2 * k + 0], v1[i * 4 * 2 + 2 * k + 1]));
      //std::cout << "appending " << v1[i * 4 * 2 + 2 * k + 0] << " " << v1[i * 4 * 2 + 2 * k + 1] << std::endl;
    }

    if (x1_max <= x1_min || y1_max <= y1_min)
      continue;
    float area1 = GetIntersection(rect1, rect1);
    //std::cout << "area1 " << area1 << std::endl;
    for (int j = 0; j < m; ++j) {

      auto x2_min = v2[j * 4 * 2 + 4 * 0 + 0];
      auto x2_max = v2[j * 4 * 2 + 4 * 0 + 0];;
      auto y2_min = v2[j * 4 * 2 + 4 * 0 + 1];;
      auto y2_max = v2[j * 4 * 2 + 4 * 0 + 1];;

      for (int k = 1; k < 4; k++) {
        x2_min = std::min(x2_min, v2[j * 4 * 2 + 2 * k + 0]);
        x2_max = std::max(x2_max, v2[j * 4 * 2 + 2 * k + 0]);
        y2_min = std::min(y2_min, v2[j * 4 * 2 + 2 * k + 1]);
        y2_max = std::max(y2_max, v2[j * 4 * 2 + 2 * k + 1]);
      }

      auto xmin = std::max(x1_min, x2_min);
      auto xmax = std::min(x1_max, x2_max);
      auto ymin = std::max(y1_min, y2_min);
      auto ymax = std::min(y1_max, y2_max);
      if ((ymax - ymin) * (xmax - xmin) <= 0)
        continue;
      for (int k = 0; k < 4; k++) {
        rect2.emplace_back(Point(v2[j * 4 * 2 + 2 * k + 0], v2[j * 4 * 2 + 2 * k + 1]));
      }
      float area2 = GetIntersection(rect2, rect2);
      float inter = GetIntersection(rect1, rect2);
      if (area2 == 0)
        continue;
      if (inter == 0)
        continue;

      output_iou[i][j] = inter / (area1 + area2 - inter);
      std::vector<Point>().swap(rect2);
    }
    std::vector<Point>().swap(rect1);
    //cout << "loop once time " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
  }
  return output_iou;

}

template <typename scalar_t>
at::Tensor bev_nms_cpu_kernel(const at::Tensor& locs, const at::Tensor& scores, const at::Tensor& classes,
                              const at::Tensor& threshold) {
  AT_ASSERTM(!locs.type().is_cuda(), "dets must be a CPU tensor");

  if (locs.numel() == 0) {
    return at::empty({0}, locs.options().dtype(at::kLong).device(at::kCPU));
  }

  int ndets = locs.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, locs.options().dtype(at::kByte).device(at::kCPU));

  at::Tensor iou = rotated_iou_kernel_v4<scalar_t>(locs, locs);

  auto suppressed = suppressed_t.data<uint8_t>();
  for (int64_t i = 0; i < ndets; i++) {
    if (suppressed[i] == 1) continue;
    scalar_t score_i = scores.data<scalar_t>()[i];
    int category = int(classes.data<scalar_t>()[i]);
    for (int64_t j = 0; j < ndets; j++) {
      if (i == j)
        continue;
      if (suppressed[j] == 1) continue;
      scalar_t iou_i_j = iou.data<scalar_t>()[i * ndets + j];
      scalar_t score_j = scores.data<scalar_t>()[j];
      if (iou_i_j > threshold.data<scalar_t>()[category] && score_i > score_j) {
        suppressed[j] = 1;
      }


    }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor rotated_iou(const at::Tensor& input1, const at::Tensor& input2) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(input1.type(), "rotated_iou", [&] {
    result = rotated_iou_kernel_v4<scalar_t>(input1, input2);
  });
  return result;
}

at::Tensor bev_nms_cpu(const at::Tensor& locs, const at::Tensor& scores, const at::Tensor& classes,
                       const at::Tensor& threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(locs.type(), "bev_nms_cpu", [&] {
    result = bev_nms_cpu_kernel<scalar_t>(locs,scores, classes, threshold);
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_nms_cpu", &bev_nms_cpu, "bev_nms_cpu");
  m.def("rotated_iou", &rotated_iou, "rotation_box iou");
}