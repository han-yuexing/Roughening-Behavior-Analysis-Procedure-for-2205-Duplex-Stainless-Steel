import math


def compute_radius():
    corr = {'50h': 180000, '4h': 14400, '30min': 1800, '2h15s': 7215, '20min': 1200, '20h': 72000, '200h': 720000,
            '1h': 3600
        , '150h': 540000, '10min5s': 605, '10min': 600, '10h': 36000, '100h': 360000}
    r1 = "./USS_Data/Unlabel_Statis"
    r2 = "./USS_Data/Unlabel"
    avg = []
    for t in os.listdir(r1):

        if t.endswith(".csv"):
            continue
        nums = 0
        radius_sum = 0
        root1 = os.path.join(r1, t)
        root2 = os.path.join(r2, t)
        for s in os.listdir(root1):
            p1 = os.path.join(root1, s)
            p2 = os.path.join(root2, s[:-4] + '.jpg')  # 用来确定比例
            image = cv2.imread(p2)
            data = pd.read_csv(p1)['perimeter']
            nums += data.shape[0]
            image = image[890:915, 700:, :]
            # 将彩色图片转换为灰度图片
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 创建一个LSD对象
            lsd = cv2.createLineSegmentDetector(0)
            # 执行检测结果
            dlines = lsd.detect(img)
            # 绘制检测结果
            X = []
            for dline in dlines[0]:
                x0 = int(round(dline[0][0]))
                y0 = int(round(dline[0][1]))
                x1 = int(round(dline[0][2]))
                y1 = int(round(dline[0][3]))
                if x0 != x1:
                    continue
                X.append(x0)
            #             cv2.line(image, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)
            X = sorted(X)
            X_co = []
            for j, x in enumerate(X):
                if j == 0:
                    if abs(X[1] - x) <= 5:
                        X_co.append(x + 2)
                    else:
                        X_co.append(x)
                    continue
                if j == len(X) - 1:
                    if abs(X[j - 1] - x) > 5:
                        X_co.append(x)
                    continue
                if abs(X[j + 1] - x) <= 5:
                    X_co.append(x + 2)
                elif abs(X[j - 1] - x) > 5:
                    X_co.append(x)
            pixels_o = int((max(X_co) - min(X_co)) / 10)

            if abs(pixels_o - 50) < 5:
                if t == "20h" and s == '2.csv':
                    rate = 2E-8
                else:
                    rate = 1E-7
            elif abs(pixels_o - 40) < 5:
                if t == '10min' and s == '1_m01.csv':
                    rate = 2.5E-8
                else:
                    rate = 0.5E-7
            elif abs(pixels_o - 30) < 5:
                if t == '10min5s':
                    rate = (5E-8) / 3
                else:
                    rate = (1E-7) / 3
            radius_sum += np.sum(data) * rate / (2 * math.pi)
        avg.append([corr[t], radius_sum / nums, nums])
    print(avg)
    pdf = pd.DataFrame(avg, columns=['time', 'avg_radius', 'numbers'])
    pdf.to_csv(os.path.join("./USS_Data/Unlabel_Statis/Summary.csv"))


if __name__ == '__main__':
    compute_radius()
