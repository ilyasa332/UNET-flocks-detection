def optical_flow(frames, winsize=15):
    prev_frame = frames[0]
    results = []
    for frame in frames[1:]:
        mask = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        mask[..., 1] = 255

        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                            None,
                                            0.5, 3, winsize, 3, 5, 1.2, 0)
        results.append(flow)
        prev_frame = frame
    return results


def optical_flow_frames(frames, winsize=15):
    flows = optical_flow(frames, winsize=winsize)
    results = []
    mask = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)
    mask[..., 1] = 255
    for flow in flows:
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        results.append(rgb)
    return results


# probs_to_write = (probs * 255).cpu().numpy().squeeze()[:-5].astype(np.uint8)
# preds_to_write = (predict.cpu().numpy().squeeze()[:-5]).astype(np.uint8)
# of = optical_flow(probs_to_write)
#
# from scipy.ndimage import gaussian_filter
#
# flows = np.stack(of)
# new_flow = gaussian_filter(flows[..., 0], sigma=1), gaussian_filter(flows[..., 1], sigma=1, radius=5)
# new_flow = np.stack(new_flow, axis=-1)
#
# figsize = 50
# step_size = 2
# results = []
# kernel_size = 25
# kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
#
# for p0, p1, flow in tqdm(zip(probs_to_write[:-1], probs_to_write[1:], of), total=len(probs_to_write)):
#     flow_new = -flow
#     flow_new[..., 0] += np.arange(256)
#     flow_new[..., 1] += np.arange(256)[:, None]
#     p0_warped = cv2.remap(p0, flow_new, None, cv2.INTER_LINEAR)
#     for im in (p0, p0_warped, p1):
#         plt.figure(figsize=(10, 10))
#         # plt.imshow(np.concatenate((p0,  p0_warped, p1), axis=1), cmap='gray')
#         plt.imshow(im, cmap='gray')
#         plt.axis('off')
#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#         plt.show()
#     break



# x, y = np.meshgrid(np.arange(0, 256, step_size), np.arange(0, 256, step_size))
# for p, flow, flow_filtered in tqdm(zip(probs_to_write[1:], of, new_flow[1:]), total=len(probs_to_write)):
#     fig = plt.figure(figsize=(figsize, figsize))
#     plt.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.imshow(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR))
#     # plt.quiver(x, y, flow[::step_size, ::step_size, 0], -flow[::step_size, ::step_size, 1], color="green", alpha=0.5)
#
#     # fig = plt.figure(figsize=(figsize, figsize))
#     # plt.axis('off')
#     # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     # plt.imshow(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR))
#     plt.quiver(x, y, flow_filtered[::step_size, ::step_size, 0], -flow_filtered[::step_size, ::step_size, 1], color="green", alpha=0.5)
#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     results.append(data)