using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using CanCanNeedJJBong.Yolo.Core.Basic;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core;

public static class YoloHelper
{
    /// <summary>
    /// 置信度排序
    /// </summary>
    /// <param name="data"></param>
    public static void RankConfidenceDegree(List<YoloData> data)
    {
        if (data == null || data.Count == 0)
        {
            return;
        }

        if (data[0].BasicData.Length == 2)
        {
            // 使用内置的排序算法，根据 BasicData[0] 降序排序
            data.Sort((x, y) => y.BasicData[0].CompareTo(x.BasicData[0]));
        }
        else
        {
            // 使用内置的排序算法，根据 BasicData[4] 降序排序
            data.Sort((x, y) => y.BasicData[4].CompareTo(x.BasicData[4]));
        }
    }

    /// <summary>
    /// nms过滤
    /// </summary>
    /// <param name="filterDataList">首次过滤数组</param>
    /// <param name="iouThreshold">iou阈值</param>
    /// <param name="allIou">全局iou</param>
    /// <returns>经过nms过滤后的Yolo数据列表</returns>
    public static List<YoloData> NMSFilter(List<YoloData> filterDataList, float iouThreshold, bool allIou)
    {
        // 对数据按置信度降序排序
        YoloHelper.RankConfidenceDegree(filterDataList);

        // 保存过滤后的结果
        List<YoloData> result = new List<YoloData>();

        foreach (var data in filterDataList)
        {
            bool isFilter = true;

            foreach (var res in result)
            {
                // 判断是否计算全局iou或属于同一类
                if (allIou || data.BasicData[5] == res.BasicData[5])
                {
                    // 计算两个矩形的交并比
                    float iou = CalcCrossMergeRatio(data.BasicData, res.BasicData);

                    // 如果iou超过阈值，设置isFilter为false
                    if (iou > iouThreshold)
                    {
                        isFilter = false;
                        break;
                    }
                }
            }

            // 如果通过过滤，将数据添加到结果中
            if (isFilter) result.Add(data);
        }

        return result;
    }

    /// <summary>
    /// 还原掩膜
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="output1">输出张量</param>
    public static void ReductionMask(List<YoloData> data, Tensor<float>? output1, int semanticSegmentationWidth,
        int[] outputTensorInfo2_Segmentation,float maskScaleRatioW,float maskScaleRatioH,float scaleRatio)
    {
        // 将输出张量转换为矩阵
        Mat ot1 = new Mat(semanticSegmentationWidth,
            outputTensorInfo2_Segmentation[2] * outputTensorInfo2_Segmentation[3], MatType.CV_32F, output1.ToArray());

        foreach (var yoloData in data)
        {
            // 原始mask
            Mat initMask = yoloData.MaskData * ot1;

            // 对mask进行并行处理，应用Sigmoid函数
            Parallel.For(0, initMask.Cols,
                col => { initMask.At<float>(0, col) = Sigmoid(initMask.At<float>(0, col)); });

            // 重塑mask
            Mat newMask = initMask.Reshape(1, outputTensorInfo2_Segmentation[2], outputTensorInfo2_Segmentation[3]);

            // 计算掩膜的左上角和宽高
            int maskX1 = Math.Max(0, (int)((yoloData.BasicData[0] - yoloData.BasicData[2] / 2) * maskScaleRatioW));
            int maskY1 = Math.Max(0, (int)((yoloData.BasicData[1] - yoloData.BasicData[3] / 2) * maskScaleRatioH));
            int maskWidth = (int)(yoloData.BasicData[2] * maskScaleRatioW);
            int maskHeight = (int)(yoloData.BasicData[3] * maskScaleRatioH);

            // 限制宽高在输出张量的范围内
            if (maskX1 + maskWidth > outputTensorInfo2_Segmentation[3])
                maskWidth = outputTensorInfo2_Segmentation[3] - maskX1;
            if (maskY1 + maskHeight > outputTensorInfo2_Segmentation[2])
                maskHeight = outputTensorInfo2_Segmentation[2] - maskY1;

            // 定义裁剪区域
            Rect rect = new Rect(maskX1, maskY1, maskWidth, maskHeight);

            // 裁剪后的掩膜
            Mat afterMat = new Mat(newMask, rect);

            // 还原原图掩膜
            Mat restoredMask = new Mat();

            // 计算放大后的宽高
            int enlargedWidth = (int)(afterMat.Width / maskScaleRatioW / scaleRatio);
            int enlargedHeight = (int)(afterMat.Height / maskScaleRatioH / scaleRatio);

            // 调整掩膜大小
            Cv2.Resize(afterMat, restoredMask, new OpenCvSharp.Size(enlargedWidth, enlargedHeight));
            // 阈值处理
            Cv2.Threshold(restoredMask, restoredMask, 0.5, 1, ThresholdTypes.Binary);

            // 将还原后的掩膜赋值回数据
            yoloData.MaskData = restoredMask;
        }
    }
    
    private static float Sigmoid(float value)
    {
        return 1 / (1 + (float)Math.Exp(-value));
    }

    #region 计算交并比 CalcCrossMergeRatio

    /// <summary>
    /// 计算交并比（IoU）
    /// </summary>
    /// <param name="rectangle1">矩形1</param>
    /// <param name="rectangle2">矩形2</param>
    /// <returns>交并比（IoU）</returns>
    private static float CalcCrossMergeRatio(float[] rectangle1, float[] rectangle2)
    {
        // 计算矩形1的边界
        float[] rect1Bounds = CalculateBounds(rectangle1);
        // 计算矩形2的边界
        float[] rect2Bounds = CalculateBounds(rectangle2);

        // 计算交集区域
        float intersectionArea = CalculateIntersectionArea(rect1Bounds, rect2Bounds);

        // 计算矩形1和矩形2的面积
        float area1 = CalculateArea(rect1Bounds);
        float area2 = CalculateArea(rect2Bounds);

        // 计算并集区域
        float unionArea = area1 + area2 - intersectionArea;

        // 返回交并比（IoU）
        return intersectionArea / unionArea;
    }

    /// <summary>
    /// 计算矩形的边界
    /// </summary>
    /// <param name="rectangle">矩形</param>
    /// <returns>矩形的边界数组 [left, top, right, bottom]</returns>
    private static float[] CalculateBounds(float[] rectangle)
    {
        return new float[]
        {
            rectangle[0] - rectangle[2] / 2, // left
            rectangle[1] - rectangle[3] / 2, // top
            rectangle[0] + rectangle[2] / 2, // right
            rectangle[1] + rectangle[3] / 2 // bottom
        };
    }

    /// <summary>
    /// 计算两个矩形的交集面积
    /// </summary>
    /// <param name="rect1">矩形1的边界</param>
    /// <param name="rect2">矩形2的边界</param>
    /// <returns>交集面积</returns>
    private static float CalculateIntersectionArea(float[] rect1, float[] rect2)
    {
        float leftBoundary = Math.Max(rect1[0], rect2[0]);
        float topBoundary = Math.Max(rect1[1], rect2[1]);
        float rightBoundary = Math.Min(rect1[2], rect2[2]);
        float bottomBoundary = Math.Min(rect1[3], rect2[3]);

        if (leftBoundary < rightBoundary && topBoundary < bottomBoundary)
        {
            return (rightBoundary - leftBoundary) * (bottomBoundary - topBoundary);
        }

        return 0;
    }

    /// <summary>
    /// 计算矩形的面积
    /// </summary>
    /// <param name="bounds">矩形的边界</param>
    /// <returns>矩形的面积</returns>
    private static float CalculateArea(float[] bounds)
    {
        return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
    }

    #endregion
    
    #region 图片缩放
    /// <summary>
    /// 图片缩放
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <returns>缩放后的图片</returns>
    public static Bitmap PictureZoom(Bitmap imgData,YoloConfig config)
    {
        // 缩放图片的目标宽度和高度
        float targetWidth = config.InferenceImageWidth;
        float targetHeight = config.InferenceImageHeight;

        // 计算缩放比例
        float scaleRatio = CalculateScaleRatio(targetWidth, targetHeight,config);

        // 按比例调整目标宽度和高度
        targetWidth *= scaleRatio;
        targetHeight *= scaleRatio;

        // 创建缩放后的图片
        Bitmap zoomImg = new Bitmap((int)targetWidth, (int)targetHeight);

        // 绘制缩放后的图片
        using (Graphics graphics = Graphics.FromImage(zoomImg))
        {
            // 设置插值模式为高质量双三次插值
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            graphics.DrawImage(imgData, 0, 0, targetWidth, targetHeight);
        }

        return zoomImg;
    }

    /// <summary>
    /// 计算缩放比例
    /// </summary>
    /// <param name="targetWidth">目标宽度</param>
    /// <param name="targetHeight">目标高度</param>
    /// <returns>缩放比例</returns>
    private static float CalculateScaleRatio(float targetWidth, float targetHeight,YoloConfig config)
    {
        if (targetWidth > config.TensorWidth || targetHeight >config.TensorHeight)
        {
            // 选择较小的缩放比例
            return Math.Min(config.TensorWidth / targetWidth, config.TensorHeight / targetHeight);
        }
        return 1f;
    }
    
    /// <summary>
    /// 图片写到张量_内存并行
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="inputTensorInfo">输入张量信息</param>
    /// <returns></returns>
    public static DenseTensor<float> PicWriTensor_Memory(Bitmap imgData, int[] inputTensorInfo)
    {
        int height = imgData.Height;
        int width = imgData.Width;
        
        // 内存图片数据
        BitmapData memoryImgData = imgData.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);
        
        int stride = memoryImgData.Stride;
        IntPtr scan0 = memoryImgData.Scan0;
        
        // 临时数据
        float[,,] tempData = new float[inputTensorInfo[1], inputTensorInfo[2], inputTensorInfo[3]];
        
        try
        {
            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    IntPtr pixel = IntPtr.Add(scan0, y * stride + x * 3);
                    tempData[2, y, x] = Marshal.ReadByte(pixel) / 255f;
                    pixel = IntPtr.Add(pixel, 1);
                    tempData[1, y, x] = Marshal.ReadByte(pixel) / 255f;
                    pixel = IntPtr.Add(pixel, 1);
                    tempData[0, y, x] = Marshal.ReadByte(pixel) / 255f;
                }
            });
        }
        finally
        {
            imgData.UnlockBits(memoryImgData);
        }
        
        // 展开临时数据
        float[] deploymentTempData = new float[inputTensorInfo[1] * inputTensorInfo[2] * inputTensorInfo[3]];
        Buffer.BlockCopy(tempData, 0, deploymentTempData, 0, deploymentTempData.Length * 4);
        
        return new DenseTensor<float>(deploymentTempData, inputTensorInfo);
    }
    
    /// <summary>
    /// 无插值写入张量
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="inputTensorInfo">输入张量信息</param>
    /// <returns></returns>
    public static DenseTensor<float> NoInterpolationWriteTensor(Bitmap imgData,YoloConfig config)
    {
        // 内存图片数据
        BitmapData memoryPicData = imgData.LockBits(new Rectangle(0, 0, imgData.Width, imgData.Height), ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);
        
        int stride = memoryPicData.Stride;
        IntPtr scan0 = memoryPicData.Scan0;
        
        // 临时数据
        float[,,] tempData = new float[config.InputTensorInfo[1], config.InputTensorInfo[2], config.InputTensorInfo[3]];
        
        float zoomImgWidth = config.InferenceImageWidth;
        float zoomImgHeight = config.InferenceImageHeight;
        
        if (zoomImgWidth > config.TensorWidth || zoomImgHeight > config.TensorHeight)
        {
            config.ScaleRatio = (config.TensorWidth / zoomImgWidth) < (config.TensorHeight / zoomImgHeight) ? (config.TensorWidth / zoomImgWidth) : (config.TensorHeight / zoomImgHeight);
            zoomImgWidth = zoomImgWidth * config.ScaleRatio;
            zoomImgHeight = zoomImgHeight * config.ScaleRatio;
        }
        
        // x,y坐标
        int xCoordinate, yCoordinate;
        
        // 系数
        float coefficient = 1 / config.ScaleRatio;
        
        for (int y = 0; y < (int)zoomImgHeight; y++)
        {
            for (int x = 0; x < (int)zoomImgWidth; x++)
            {
                xCoordinate = (int)(x * coefficient);
                yCoordinate = (int)(y * coefficient);
                IntPtr pixel = IntPtr.Add(scan0, yCoordinate * stride + xCoordinate * 3);
                tempData[2, y, x] = Marshal.ReadByte(pixel) / 255f;
                pixel = IntPtr.Add(pixel, 1);
                tempData[1, y, x] = Marshal.ReadByte(pixel) / 255f;
                pixel = IntPtr.Add(pixel, 1);
                tempData[0, y, x] = Marshal.ReadByte(pixel) / 255f;
            }
        }

        imgData.UnlockBits(memoryPicData);
        float[] deploymentTempData = new float[config.InputTensorInfo[1] * config.InputTensorInfo[2] * config.InputTensorInfo[3]];
        Buffer.BlockCopy(tempData, 0, deploymentTempData, 0, deploymentTempData.Length * 4);
        return new DenseTensor<float>(deploymentTempData, config.InputTensorInfo);
    }
    
    /// <summary>
    /// 还原返回坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    public static void RestoreCoordinates(List<YoloData> data,YoloConfig config)
    {
        if (data.Count == 0) return;

        // 如果 BasicData 长度大于 2，调整坐标比例
        if (data[0].BasicData.Length > 2)
        {
            foreach (var item in data)
            {
                for (int j = 0; j < 4; j++)
                {
                    item.BasicData[j] /= config.ScaleRatio;
                }
            }
        }

        // 如果 PointKeys 不为空，调整坐标比例
        if (data[0].PointKeys != null)
        {
            foreach (var item in data)
            {
                foreach (var point in item.PointKeys)
                {
                    point.X /= config.ScaleRatio;
                    point.Y /= config.ScaleRatio;
                }
            }
        }
    }
    
    /// <summary>
    /// 去除越界坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    public static void RemoveCoordinates(List<YoloData> data,YoloConfig config)
    {
        //倒序移除
        for (int i = data.Count - 1; i >= 0; i--)
        {
            if (data[i].BasicData[0] > config.InferenceImageWidth ||
                data[i].BasicData[1] > config.InferenceImageHeight ||
                data[i].BasicData[2] > config.InferenceImageWidth ||
                data[i].BasicData[3] > config.InferenceImageHeight)
            {
                data.RemoveAt(i);
            }
        }
        
    }
    
    /// <summary>
    /// 还原画图坐标
    /// </summary>
    /// <param name="list">数据列表</param>
    public static void RestoreDrawCoordinates(List<YoloData> list)
    {
        if (list.Count > 0 && list[0].BasicData.Length > 2)
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].BasicData[0] = list[i].BasicData[0] - list[i].BasicData[2] / 2;
                list[i].BasicData[1] = list[i].BasicData[1] - list[i].BasicData[3] / 2;
            }
        }
    }
    
    /// <summary>
    /// 还原中心坐标
    /// </summary>
    /// <param name="list">数据列表</param>
    public static void RestoreMidCoordinates(List<YoloData> list)
    {
        if (list.Count > 0 && list[0].BasicData.Length > 2)
        {
            for (int i = 0; i < list.Count; i++)
            {
                list[i].BasicData[0] = list[i].BasicData[0] + list[i].BasicData[2] / 2;
                list[i].BasicData[1] = list[i].BasicData[1] + list[i].BasicData[3] / 2;
            }
        }
    }
    
    /// <summary>
    /// 生成掩膜图像_内存并行
    /// </summary>
    /// <param name="matData">mat数据</param>
    /// <param name="color">颜色</param>
    /// <returns></returns>
    public static Bitmap GenMaskImg_Memory(Mat matData, Color color)
    {
        Bitmap maskImg = new Bitmap(matData.Width, matData.Height, PixelFormat.Format32bppArgb);
        BitmapData maskImgData = maskImg.LockBits(new Rectangle(0, 0, maskImg.Width, maskImg.Height), ImageLockMode.ReadWrite,
            PixelFormat.Format32bppArgb);
        int height = maskImg.Height;
        int width = maskImg.Width;
        Parallel.For(0, height, i =>
        {
            for (int j = 0; j < width; j++)
            {
                if (matData.At<float>(i, j) == 1)
                {
                    // 起始像素
                    IntPtr startPix = IntPtr.Add(maskImgData.Scan0, i * maskImgData.Stride + j * 4);
                    // 颜色信息
                    byte[] colorMessage = new byte[] { color.B, color.G, color.R, color.A };
                    Marshal.Copy(colorMessage, 0, startPix, 4);
                }
            }
        });
        maskImg.UnlockBits(maskImgData);
        return maskImg;
    }
    
    /// <summary>
    /// OBB坐标转换
    /// </summary>
    /// <param name="data">数据</param>
    /// <returns>返回OBB矩形结构,分别代表了四个点的坐标</returns>
    public static OBBRectangularStructure OBBConversion(YoloData data)
    {
        float x = data.BasicData[0];
        float y = data.BasicData[1];
        float w = data.BasicData[2];
        float h = data.BasicData[3];
        float r = data.BasicData[6];
        float cos_value = (float)Math.Cos(r);
        float sin_value = (float)Math.Sin(r);
        float[] vec1 = { w / 2 * cos_value, w / 2 * sin_value };
        float[] vec2 = { -h / 2 * sin_value, h / 2 * cos_value };
        
        OBBRectangularStructure oBBrectangle = new OBBRectangularStructure();
        oBBrectangle.pt1 = new PointF(x + vec1[0] + vec2[0], y + vec1[1] + vec2[1]);
        oBBrectangle.pt2 = new PointF(x + vec1[0] - vec2[0], y + vec1[1] - vec2[1]);
        oBBrectangle.pt3 = new PointF(x - vec1[0] - vec2[0], y - vec1[1] - vec2[1]);
        oBBrectangle.pt4 = new PointF(x - vec1[0] + vec2[0], y - vec1[1] + vec2[1]);
        return oBBrectangle;
    }
    
    #endregion
}