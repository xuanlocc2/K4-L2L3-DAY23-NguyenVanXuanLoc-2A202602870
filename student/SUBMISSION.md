# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Báo cáo dựa trên lần chạy CP5 sau khi hoàn thành Part E–H. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Văn Xuân Lộc
- MSSV: 2A202602870
- Email: nguyen.vanxuanloc1102@gmail.com
- Link repo (fork): https://github.com/xuanlocc2/K4-L2L3-DAY23-NguyenVanXuanLoc-2A202602870-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): lấy hash 40 ký tự của commit CP6 chứa báo cáo này sau khi commit/push; nộp đúng hash đó trên VLearn. Không dùng hash CP5 vì chưa chứa báo cáo hoàn chỉnh.

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame/mode), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `0.9700934579439252`, `0.7004048582995951`, `519/16/222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.15032268781360134 m`, `502`, `11.343649056695735 m²`, `0`, `239`, `2.522613065326633`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `0.1358667883353908 m`, `502`, `9.26681165463209 m²`, `0`, `239`, `2.522613065326633`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: fused giảm RMSE khoảng `0.0144559 m` (9.62%) trong khi số ghép, ghost và miss giữ nguyên. Cả hai có `precision_track = 502/(502+0) = 1`, `coverage = 502/519 = 0.9672447013487476`, nên `q = 1` theo rubric. Chênh lệch `rmse_fused − rmse_lidar = -0.0144559 m ≤ 0.05 m` đạt điều kiện nhất quán. Có 239 GT-frame chưa ghép, do đó không thể kết luận tracker theo dõi mọi xe chỉ từ RMSE thấp. Detector bỏ sót 222 GT-frame; số miss tracking còn chịu ảnh hưởng thời gian chờ xác nhận và việc duy trì track.

Nguồn số liệu: [metrics.json](artifacts/metrics.json), [metrics_lidar.json](artifacts/metrics_lidar.json), [metrics_fused.json](artifacts/metrics_fused.json). Log tổng có 398 record duy nhất, log riêng có 199 record/mode; đã kiểm tra invariant từng record, đối chiếu log riêng với log tổng và tính lại mọi metric bằng `fusion_lab.evaluation.validate_metrics_records`. Ví dụ [log LiDAR](artifacts/grade_run_lidar.log) frame 0 có 0 confirmed/2 miss; frame 4 có 2 confirmed/2 matches/0 miss; frame 198 có 3 matches/6 miss. Đây là bằng chứng RMSE phải đọc cùng độ phủ và giai đoạn xác nhận.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   LiDAR đo tâm hộp `z = (x, y, z)` trong hệ cảm biến, đơn vị mét, với `R` 3×3; `H` tuyến tính 3×6. Camera đo `z = (u, v)` theo pixel, `R` 2×2 có đường chéo `sigma_cam_i²`, `sigma_cam_j²`; `h(x)` là chiếu pinhole sau đổi tọa độ, `H` 2×6 là Jacobian do platform cung cấp. Cả hai dùng chung state `[px, py, pz, vx, vy, vz]ᵀ`, innovation `z − h(x)` và `S = HPHᵀ + R` trong [kalman.py](workspace/kalman.py); đo camera được tạo trong [camera_fusion.py](workspace/camera_fusion.py).
2. Vì sao cần gating Mahalanobis trước khi gán?
   `d² = gammaᵀ S⁻¹ gamma` xét độ bất định của cả track và phép đo, khác khoảng cách Euclid thuần túy: cùng residual nhưng `S` lớn sẽ có cost nhỏ hơn. [association.py](workspace/association.py) dùng `kalman.innovation`/`innovation_covariance`, tính bằng giải hệ tuyến tính và so với `chi2.ppf(gating_threshold, sensor.dim_meas)` (3 chiều LiDAR, 2 chiều camera). Cặp ngoài FOV bị loại trước khi tính residual/Jacobian; sau gating, greedy chọn cost hữu hạn nhỏ nhất và xóa hàng/cột để ghép một-một.
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   Đây là track-then-fuse: một danh sách track được predict đúng một lần/frame, update LiDAR rồi update camera nếu có FRONT. Vòng lặp trong [run_lab.py](../platform/fusion_lab/scripts/run_lab.py) gọi `KF.predict`, association LiDAR rồi association camera; không predict thêm giữa hai sensor. Log [grade_run.log](artifacts/grade_run.log) có record kết quả cuối frame cho `lidar` và `fused` (ví dụ frame 4 có 2 matches ở cả hai mode); log không ghi từng lời gọi predict/update, nên thứ tự nội bộ phải đối chiếu code runner. Các đo cùng frame dùng cùng thời gian logic `frame_index * dt`, không mô phỏng sensor async.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   Extrinsic/intrinsic sai làm `h(x)` lệch có hệ thống so với pixel đo, nên innovation camera có thể có bias theo hướng/độ sâu. Nếu `d²` vượt cổng χ², phép gán bị loại; nếu vẫn qua cổng, update có thể kéo state sai và tăng RMSE. Đổi extrinsic còn có thể làm sai FOV hoặc đặt điểm sau camera. [camera_fusion.py](workspace/camera_fusion.py) kiểm tra tọa độ hữu hạn và độ sâu `> 1e-6`; [association.py](workspace/association.py) gate trước update. Lần chạy này không chủ động làm lệch calibration, nên đây là giải thích cơ chế, không phải kết quả thí nghiệm drift.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   Danh sách đo rỗng không cho phép suy sensor từ `meas_list[0]`, nhưng vẫn cần biết lượt đang xử lý để quản lý các track chưa ghép. [association.py](workspace/association.py) luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. [manager.py](../platform/fusion_lab/tracking/manager.py) chỉ ghi hit/miss, tạo và xóa track ở lượt LiDAR; miss chỉ giảm score nếu track trong FOV. Lượt camera không gọi quy tắc score/delete và không tạo track; cặp camera được ghép chỉ update EKF. Các test `test_association_empty_pass_still_manages_tracks`, `test_empty_lidar_frame_scores_then_deletes_exhausted_track`, `test_camera_hit_refines_state_without_changing_score` kiểm tra các đường đi này.
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   [track_management.py](workspace/track_management.py) khởi tạo từ đo LiDAR: đổi vị trí sang hệ xe, vận tốc bằng 0, covariance vị trí `R` xoay sang hệ xe, covariance vận tốc từ `sigma_p44/55/66²`, score `1/window`, state `initialized`. Hit LiDAR tăng `1/window`, chặn ở 1; miss trong FOV giảm `1/window`. Xác nhận khi `score > confirmed_threshold`; hit chưa đạt ngưỡng chuyển sang `tentative`, còn track đã confirmed không bị hạ state sau miss. Xóa nếu `P[0,0] > max_P` hoặc `P[1,1] > max_P`, hoặc confirmed có `score < delete_threshold`, hoặc chưa confirmed có `score ≤ 0`. Tham số hiện tại là `window=6`, ngưỡng xác nhận `0.8`, ngưỡng xóa `0.6`, `max_P=9 m²`, đều đọc từ `get_tracking_params()`.

## Bonus (không bắt buộc)

- Không.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (CP1–CP2, theo lịch sử commit); ChatGPT/Codex (kiểm tra CP1–CP2 và thực hiện CP3–CP6).
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ cài sáu hàm EKF Part E, ba hàm mô hình camera Part G, năm hàm association Part F, ba hàm quản lý track Part H; chạy test, chạy Waymo compare, kiểm tra log/metrics và soạn báo cáo từ artifact thực tế.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): quy trình kiểm tra bằng công cụ có `test_kalman.py` 4 passed, nhóm camera 20 passed, nhóm association 9 passed và toàn bộ `student/tests` 128 passed, 0 failed/xfailed. Sau sửa code Part H, đã chạy đủ frame 0–198 ở compare seed 0 và đối chiếu cả sáu artifact bằng validator, kiểm tra cặp `(mode, frame)` duy nhất và các invariant. Các bước này do công cụ hỗ trợ chạy; người nộp cần đọc lại code và câu trả lời để tự giải thích khi vấn đáp. Không sửa test/platform hoặc sửa tay artifact. Không dùng số liệu chưa chạy.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên code cung cấp sẵn; không sửa platform/test
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP` sau commit CP6
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
