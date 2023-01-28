# Decision Tree
[**Chi tiết bài viết**](...)

**Dưới đây là nội dung tóm tắt**
### Nội dung:
1. Giới thiệu
2. Các ứng dụng
3. Thuật toán
4. Ưu/khuyết điểm
5. Xây dựng Decision Tree sử dụng sklearn

## 1. Giới thiệu
+ Decision Tree là thuật toán thuộc nhóm Supervised Learning được sử dụng cho cả **classification** và **regression**.
+ Là thuật toán theo mô hình cây được sử dụng để xác định kết quả của hành động. Mỗi nhánh cây đại diện cho một quyết định, sự xuất hiện hay phản ứng có thể xảy ra.

## 2. Các ứng dụng
+ Decision Tree được sử dụng để phát triển các mô hình dự đoán và phân loại trong nhiều lĩnh vực khác nhau như:
    + Business Management
    + Customer Relationship Management
    + Fraudulent Statement Detection
    + Engineering
    + Energy Consumption
    + Fault Diagnosis
    + Healthcare Management
    + Agriculture
    + ...

**Ví dụ:**  
    + Đánh giá các cơ hội mở rộng thương hiệu cho một doanh nghiệp sử dụng dữ liệu bán hàng lịch sử.  
    + Xác định khả năng khách hàng mua sản phẩm sử dụng dữ liệu nhân khẩu học cho phép nhắm đến mục tiêu ngân sách quảng cáo hạn chế.  
    + Dự đoán khả năng cho khách hàng vay tiền sử dụng các mô hình dự đoán được tạo ra từ dữ liệu lịch sử.  
    + Giúp ưu tiên điều trị bệnh nhân trong phòng cấp cứu bằng mô hình dự đoán dựa trên các yếu tố như tuổi, huyết áp, giới tính, vị trí và mức độ nghiêm trọng của cơn đau và các phép đo khác.

+ Ngoài ra, Decision Tree thường được sử dụng trong nghiên cứu hoạt động, đặc biệt trong phân tích quyết định, giúp xác định chiến lược có khả năng đạt được mục tiêu tốt nhất.

## 3. Thuật toán
### Ý tưởng
+ Tìm các tính năng (feature) mô tả có chứa "thông tin" nhất về target feature và sau đó chia tập dữ liệu dọc theo các giá trị của các feature này sao cho các giá trị target feature cho các tập con (sub_dataset) càng "thuần khiết" càng tốt.

=> Các tính năng mô tả dẫn đến target feature "thuần khiết" nhất được gọi là có "thông tin" nhất.

+ Quá trình tìm kiếm các tính năng có thông tin nhất ("most infor mative") được thực hiện cho đến khi chúng ta hoàn thành điều kiện dừng khi cuối cùng kết thúc ở nút lá (**leaf node**). Các nút lá chứa các thông tin dự đoán mà chúng ta sẽ thực hiện cho các thực thể mới được trình bày cho trained model.
+ Điều này khả thi vì model đã học được cấu trúc cơ bản của dữ liệu huấn luyện (training data) và do đó có thể đưa ra một số giả định, đưa ra các dự đoán về giá trị target feature (class) của các thực thể chưa biết.

### Xây dựng cây quyết định
+ Bắt đầu với tất cả các sample tại một node.
+ Các sample phân vùng dựa trên input để tạo tập con thuần khiết nhất (purest subset).
+ Lặp lại quá trình phân vùng dữ liệu vào các tập con thuần khiết hơn.

### GINI
+ Làm việc với categorical target variable (ví dụ: "Success" hay "Failure", "Pass" hay "Fail")
+ Thực hiện chia cây theo Binary splits (2 nhánh).
+ Gini index càng thấp thì tính đồng nhất càng cao.
+ CART (Classification and Regression Tree) sử dụng Gini method để tạo binary splits.

$$Gini = 1 - \sum_{i=1}^c(P_i)^2 $$

+ Tính Gini cho từng thuộc tính, theo class kết quả.
+ Thuộc tính nào trong nhóm các thuộc tính có gini bé nhất => chọn thuộc tính đó để chia nhánh cho cây.

**Tính Gini index cho thuộc tính**
+ Làm tương tự cho các thuộc tính còn lại.
+ So sánh gini của các thuộc tính => chọn thuộc tính có gini nhỏ nhất để chia cây.

## Entropy và Information Gain
**Entropy**
+ Entropy = 0: mẫu hoàn toàn thuần khiết.
+ Entropy = 1: mẫu không thuần khiết (trộn đều).
+ Tuỳ vào số lượng các class trong dataset, entropy có thể lớn hơn 1, tuy nhiên ý nghĩa như nhau: Entropy càng lớn thì càng ít thuần khiết.

**Information Gain**
+ Dựa vào việc giảm entropy sau khi dataset được phân chia dựa trên một thuộc tính.
+ Việc xây dựng decision tree dựa vào việc tìm thuộc tính trả về information gain cao nhất (các nhánh đồng nhất nhất).

$$Entropy = \sum_{i=1}^c - p_i*log_2(p_i)$$

Hoặc: 

$$Entropy = -\sum_{i=1}^cp_i*log_2(p_i)$$

$$Gain(T, X) = Entropy(T) - Entropy(T, X)$$

+ Tính Gain cho từng thuộc tính, theo class kết quả.
+ Thuộc tính nào trong nhóm các thuộc tính có Gain lớn nhất => chọn thuộc tính đó để chia nhánh cho cây.

**Khi nào dừng chia nhỏ node?**
+ Khi tất cả các sample có class label.
+ Số lượng các sample trong node đạt đến mức tối thiểu.
+ Thay đổi trong đo lường độ không đồng nhất nhỏ hơn ngưỡng (threshold).
+ Đạt được độ sâu cây tối đa.
+ ...

**Decision Tree trong Classification**
+ Cây kết quả thường đơn giản và dễ diễn giải.
+ Tính toán không tốn kém.
+ Dùng ranh giới quyết định là Rectilinear.

## 4. Ưu/khuyết điểm
### Ưu điểm
+ Dễ hiểu, dễ xây dựng mô hình.
+ Không cần chuẩn hoá tính năng.
+ Có thể áp dụng trong cả Classification và Regression.
+ Có thể mô hình các quan hệ phi tuyến (non-linear relationship).
+ Có thể mô hình các tương tác giữa các tính năng mô tả khác nhau.

### Khuyết điểm
+ Nếu tính năng liên tục (continuous feature) được sử dụng thì cây có thể trở nên rất lớn và ít diễn giải.
+ Những thay đổi nhỏ trong dữ liệu có thể dẫn đến cây hoàn toàn khác (**có thể dùng random forest để khắc phục**).
+ Nếu số lượng tính năng tương đối lớn àm số lượng mẫu lại nhỏ có thể dẫn đến không phù hợp dữ liệu.

## 5. Xây dựng Decision Tree sử dụng sklearn
### Xây dựng cây quyết định sử dụng sklearn
+ `sklearn.tree.DecisionTreeClassifier`: Đây là một mô hình phân loại dựa trên cây quyết định rất mạnh mẽ giúp chúng ta có thể thực hiện một tree model nhanh hơn, hiệu quả hơn và gọn gàng hơn chỉ vài dòng lệnh.
+ `sklearn.tree.DecisionTreeRegressor`: Đây là một mô hình dự đoán numeric dựa trên cây quyết định.
