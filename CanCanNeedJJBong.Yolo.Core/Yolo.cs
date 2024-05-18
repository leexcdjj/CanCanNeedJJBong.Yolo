using System.Collections.Concurrent;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using CanCanNeedJJBong.Yolo.Core.Basic;
using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy.YoloV8;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CanCanNeedJJBong.Yolo.Core;

public class Yolo
{
    /// <summary>
    /// 模型会话
    /// </summary>
    public InferenceSession ModelSession { get; set; }

    /// <summary>
    /// 张量宽度
    /// </summary>
    public int TensorWidth { get; set; }

    /// <summary>
    /// 张量高度
    /// </summary>
    public int TensorHeight { get; set; }

    /// <summary>
    /// 模型输入名
    /// </summary>
    public string ModelInputName { get; set; }

    /// <summary>
    /// 模型输出名
    /// </summary>summary
    public string ModelOutputName { get; set; }

    /// <summary>
    /// 输入张量信息
    /// </summary>
    public int[] InputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息
    /// </summary>
    public int[] OutputTensorInfo { get; set; }

    /// <summary>
    /// 输出张量信息2_分割
    /// </summary>
    public int[] OutputTensorInfo2_Segmentation { get; set; }

    /// <summary>
    /// 推理图片宽度
    /// </summary>
    public int InferenceImageWidth { get; set; }

    /// <summary>
    /// 推理图片高度
    /// </summary>
    public int InferenceImageHeight { get; set; }

    /// <summary>
    /// 输入张量
    /// </summary>
    public DenseTensor<float> InputTensor { get; set; }

    /// <summary>
    /// YOLO版本
    /// </summary>
    public int YoloVersion { get; set; }

    /// <summary>
    /// mask缩放比例W
    /// </summary>
    public float MaskScaleRatioW { get; set; } = 0;

    /// <summary>
    /// mask缩放比例H
    /// </summary>
    public float MaskScaleRatioH { get; set; } = 0;

    /// <summary>
    /// 模型版本
    /// </summary>
    public string ModelVersion { get; set; } = "";

    /// <summary>
    /// 任务类型
    /// </summary>
    public string TaskType { get; set; } = "";

    /// <summary>
    /// 语义分割宽度
    /// </summary>
    public int SemanticSegmentationWidth { get; set; } = 0;

    /// <summary>
    /// 动作宽度
    /// </summary>
    public int ActionWidth { get; set; } = 0;

    /// <summary>
    /// 缩放比例
    /// </summary>
    public float ScaleRatio { get; set; } = 1;

    /// <summary>
    /// 标签组
    /// </summary>
    public string[] LabelGroup { get; set; }

    /// <summary>
    /// 执行任务模式
    /// </summary>
    private int ExecutionTaskMode = 0;

    /// <summary>
    /// 任务模式
    /// </summary>
    public int TaskMode
    {
        get { return ExecutionTaskMode; }
        set
        {
            if (TaskType == "classify")
            {
                ExecutionTaskMode = 0;
            }
            //检测
            else if (TaskType == "detect")
            {
                ExecutionTaskMode = 1;
            }
            //分割
            else if (TaskType == "segment")
            {
                if (value == 1 || value == 2 || value == 3)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 3;
                }
            }
            //动作
            else if (TaskType == "pose")
            {
                if (value == 1 || value == 4 || value == 5)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 5;
                }
            }
            else if (TaskType == "obb")
            {
                if (value == 6)
                {
                    ExecutionTaskMode = value;
                }
                else
                {
                    ExecutionTaskMode = 6;
                }
            }
            else
            {
                //如果什么都数据都没有,默认是检测模型
                ExecutionTaskMode = 1;
            }
        }
    }

    /// <summary>
    /// 推理主要函数
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="confidenceDegree">置信度:0-1的小数,数值越高越快,检测精度要求也会越高.</param>
    /// <param name="iouThreshold">iou阈值:0-1的小数,数值越高出现重复框的概率就越大,该值代表允许方框的最大重复面积</param>
    /// <param name="allIou">全局iou:false代表不同种类的框按允许重叠,true代表所有不同标签种类的框都要按照iou阈值不准重叠</param>  
    /// <param name="preprocessingMode">预处理模式:0表示高精度模式,对小物体有更高精度的检测;1表示高速模式,尤其是对大图像,大幅提高推理速度</param>   
    /// <returns>返回列表的基本数据格式{目标中心点x,目标中心点y,识别宽度,识别高度,置信度,标签索引}</returns>
    public List<YoloData> ModelReasoning(Bitmap imgData, float confidenceDegree = 0.5f, float iouThreshold = 0.3f, bool allIou = false, int preprocessingMode = 1)
    {
        InputTensor = new DenseTensor<float>(InputTensorInfo);
        var sp = InputTensor.Buffer.Span;
        ScaleRatio = 1;
        InferenceImageWidth = imgData.Width;
        InferenceImageHeight = imgData.Height;
        
        if (preprocessingMode == 0)
        {
            imgData = PictureZoom(imgData);
            InputTensor = PicWriTensor_Memory(imgData, InputTensorInfo);
        }
        if (preprocessingMode == 1)
        {
            InputTensor = NoInterpolationWriteTensor(imgData, InputTensorInfo);
        }
        
        //容器
        IReadOnlyCollection<NamedOnnxValue> container = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(ModelInputName, InputTensor)
        };

        Tensor<float> output0;
        Tensor<float> output1;
        
        // 过滤数据数组
        List<YoloData> filterDataList = new List<YoloData>();
        List<YoloData> result = new List<YoloData>();
        
        if (ExecutionTaskMode == 0)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();

            result = new ClassifyInferenceStrategy().ExecuteTask(output0,confidenceDegree,iouThreshold,allIou,this);
        }
        
        if (ExecutionTaskMode == 1)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();
            if (YoloVersion == 8)
            {
                filterDataList = new DetectInferenceStrategy().ExecuteTask(output0,confidenceDegree,iouThreshold,allIou,this);
            }
            else if (YoloVersion == 5)
            {
                // todo：暂不支持yolov5
                // filterDataList = 置信度过滤_yolo5检测(output0, confidenceDegree);
            }
            else
            {
                // todo：暂不支持yolov6
                // filterDataList = 置信度过滤_yolo6检测(output0, confidenceDegree);
            }

            result = YoloHelper.NMSFilter(filterDataList, iouThreshold, allIou);
        }
        
        if (ExecutionTaskMode == 2 || ExecutionTaskMode == 3)
        {
            // 返回数据
            var list = ModelSession.Run(container);
            output0 = list.First().AsTensor<float>();
            output1 = list.ElementAtOrDefault(1)?.AsTensor<float>();
            if (YoloVersion == 8)
            {
                filterDataList = new SegmentInferenceStrategy().ExecuteTask(output0, confidenceDegree,iouThreshold,allIou,this);
                // filterDataList = ConfidenceFilter_YoloV8_Split(output0, confidenceDegree);
            }
            else
            {
                // todo: 暂不支持yolov5分割
                // filterDataList = 置信度过滤_yolo5分割(output0, confidenceDegree);
            }

            result = YoloHelper.NMSFilter(filterDataList, iouThreshold, allIou);
            
            ReductionMask(result, output1);
        }
        
        if (ExecutionTaskMode == 4 || ExecutionTaskMode == 5)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();

            filterDataList = new PoseInferenceStrategy().ExecuteTask(output0,confidenceDegree,iouThreshold,allIou,this);
            // filterDataList = ConfidenceFilter_Action(output0, confidenceDegree);
            
            result = YoloHelper.NMSFilter(filterDataList, iouThreshold, allIou);
        }
        
        if (ExecutionTaskMode == 6)
        {
            output0 = ModelSession.Run(container).First().AsTensor<float>();

            filterDataList = new ObbInferenceStrategy().ExecuteTask(output0, confidenceDegree,iouThreshold,allIou,this);
            // filterDataList = ConfidenceFilter_OBB(output0, confidenceDegree);
            
            result = YoloHelper.NMSFilter(filterDataList, iouThreshold, allIou);
        }

        RestoreCoordinates(result);
        if (ExecutionTaskMode != 0)
        {
            RemoveCoordinates(result);
        }

        return result;
    }

    #region 图片缩放
    /// <summary>
    /// 图片缩放
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <returns>缩放后的图片</returns>
    private Bitmap PictureZoom(Bitmap imgData)
    {
        // 缩放图片的目标宽度和高度
        float targetWidth = InferenceImageWidth;
        float targetHeight = InferenceImageHeight;

        // 计算缩放比例
        float scaleRatio = CalculateScaleRatio(targetWidth, targetHeight);

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
    private float CalculateScaleRatio(float targetWidth, float targetHeight)
    {
        if (targetWidth > TensorWidth || targetHeight > TensorHeight)
        {
            // 选择较小的缩放比例
            return Math.Min(TensorWidth / targetWidth, TensorHeight / targetHeight);
        }
        return 1f;
    }
    
    #endregion
    
    
    /// <summary>
    /// 图片写到张量_内存并行
    /// </summary>
    /// <param name="imgData">图片数据</param>
    /// <param name="inputTensorInfo">输入张量信息</param>
    /// <returns></returns>
    private DenseTensor<float> PicWriTensor_Memory(Bitmap imgData, int[] inputTensorInfo)
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
    private DenseTensor<float> NoInterpolationWriteTensor(Bitmap imgData, int[] inputTensorInfo)
    {
        // 内存图片数据
        BitmapData memoryPicData = imgData.LockBits(new Rectangle(0, 0, imgData.Width, imgData.Height), ImageLockMode.ReadOnly,
            PixelFormat.Format24bppRgb);
        
        int stride = memoryPicData.Stride;
        IntPtr scan0 = memoryPicData.Scan0;
        
        // 临时数据
        float[,,] tempData = new float[inputTensorInfo[1], inputTensorInfo[2], inputTensorInfo[3]];
        
        float zoomImgWidth = InferenceImageWidth;
        float zoomImgHeight = InferenceImageHeight;
        
        if (zoomImgWidth > TensorWidth || zoomImgHeight > TensorHeight)
        {
            ScaleRatio = (TensorWidth / zoomImgWidth) < (TensorHeight / zoomImgHeight) ? (TensorWidth / zoomImgWidth) : (TensorHeight / zoomImgHeight);
            zoomImgWidth = zoomImgWidth * ScaleRatio;
            zoomImgHeight = zoomImgHeight * ScaleRatio;
        }
        
        // x,y坐标
        int xCoordinate, yCoordinate;
        
        // 系数
        float coefficient = 1 / ScaleRatio;
        
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
        float[] deploymentTempData = new float[inputTensorInfo[1] * inputTensorInfo[2] * inputTensorInfo[3]];
        Buffer.BlockCopy(tempData, 0, deploymentTempData, 0, deploymentTempData.Length * 4);
        return new DenseTensor<float>(deploymentTempData, inputTensorInfo);
    }
    
    /// <summary>
    /// 还原掩膜
    /// </summary>
    /// <param name="data">数据</param>
    /// <param name="output1">输出张量</param>
    private void ReductionMask(List<YoloData> data, Tensor<float>? output1)
    {
        // 将输出张量转换为矩阵
        Mat ot1 = new Mat(SemanticSegmentationWidth,
            OutputTensorInfo2_Segmentation[2] * OutputTensorInfo2_Segmentation[3], MatType.CV_32F, output1.ToArray());

        foreach (var yoloData in data)
        {
            // 原始mask
            Mat initMask = yoloData.MaskData * ot1;

            // 对mask进行并行处理，应用Sigmoid函数
            Parallel.For(0, initMask.Cols,
                col => { initMask.At<float>(0, col) = Sigmoid(initMask.At<float>(0, col)); });

            // 重塑mask
            Mat newMask = initMask.Reshape(1, OutputTensorInfo2_Segmentation[2], OutputTensorInfo2_Segmentation[3]);

            // 计算掩膜的左上角和宽高
            int maskX1 = Math.Max(0, (int)((yoloData.BasicData[0] - yoloData.BasicData[2] / 2) * MaskScaleRatioW));
            int maskY1 = Math.Max(0, (int)((yoloData.BasicData[1] - yoloData.BasicData[3] / 2) * MaskScaleRatioH));
            int maskWidth = (int)(yoloData.BasicData[2] * MaskScaleRatioW);
            int maskHeight = (int)(yoloData.BasicData[3] * MaskScaleRatioH);

            // 限制宽高在输出张量的范围内
            if (maskX1 + maskWidth > OutputTensorInfo2_Segmentation[3])
                maskWidth = OutputTensorInfo2_Segmentation[3] - maskX1;
            if (maskY1 + maskHeight > OutputTensorInfo2_Segmentation[2])
                maskHeight = OutputTensorInfo2_Segmentation[2] - maskY1;

            // 定义裁剪区域
            Rect rect = new Rect(maskX1, maskY1, maskWidth, maskHeight);

            // 裁剪后的掩膜
            Mat afterMat = new Mat(newMask, rect);

            // 还原原图掩膜
            Mat restoredMask = new Mat();

            // 计算放大后的宽高
            int enlargedWidth = (int)(afterMat.Width / MaskScaleRatioW / ScaleRatio);
            int enlargedHeight = (int)(afterMat.Height / MaskScaleRatioH / ScaleRatio);

            // 调整掩膜大小
            Cv2.Resize(afterMat, restoredMask, new OpenCvSharp.Size(enlargedWidth, enlargedHeight));
            // 阈值处理
            Cv2.Threshold(restoredMask, restoredMask, 0.5, 1, ThresholdTypes.Binary);

            // 将还原后的掩膜赋值回数据
            yoloData.MaskData = restoredMask;
        }
    }
    
    private float Sigmoid(float value)
    {
        return 1 / (1 + (float)Math.Exp(-value));
    }
    
    /// <summary>
    /// 还原返回坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    private void RestoreCoordinates(List<YoloData> data)
    {
        if (data.Count == 0) return;

        // 如果 BasicData 长度大于 2，调整坐标比例
        if (data[0].BasicData.Length > 2)
        {
            foreach (var item in data)
            {
                for (int j = 0; j < 4; j++)
                {
                    item.BasicData[j] /= ScaleRatio;
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
                    point.X /= ScaleRatio;
                    point.Y /= ScaleRatio;
                }
            }
        }
    }

    /// <summary>
    /// 去除越界坐标
    /// </summary>
    /// <param name="data">数据列表</param>
    private void RemoveCoordinates(List<YoloData> data)
    {
        // 倒序移除越界数据
        for (int i = data.Count - 1; i >= 0; i--)
        {
            // 如果数据的任何一个坐标超出张量范围，移除该数据
            if (data[i].BasicData[0] > TensorWidth ||
                data[i].BasicData[1] > TensorHeight ||
                data[i].BasicData[2] > TensorWidth ||
                data[i].BasicData[3] > TensorHeight)
            {
                data.RemoveAt(i);
            }
        }
    }
}