using System.Drawing;
using CanCanNeedJJBong.Yolo.Core.Basic;
using CanCanNeedJJBong.Yolo.Core.Builder;
using CanCanNeedJJBong.Yolo.Core.TaskModelStrategy;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CanCanNeedJJBong.Yolo.Core;

public class YoloService
{
    private YoloConfig Config { get; set; }
    private ITaskModelInferenceStrategyFactory TaskModelInferenceStrategyFactory { get; set; }

    public YoloService(string modelPath, bool enableGpu, int gpuIndex,int yoloVersion, int tensorWidth, int tensorHeight)
    {
        enableGpu = false;
        gpuIndex = 0;
        YoloConfigBuilder builder = new YoloConfigBuilder();
        Config = builder.SetModelSession(modelPath, enableGpu, gpuIndex)
            .SetModelMetadata()
            .SetInferenceParameters(yoloVersion, tensorWidth, tensorHeight)
            .GetYoloConfig();

        TaskModelInferenceStrategyFactory = new TaskModelInferenceStrategyFactory();
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
        Config.InputTensor = new DenseTensor<float>(Config.InputTensorInfo);
        var sp = Config.InputTensor.Buffer.Span;
        Config.ScaleRatio = 1;
        Config.InferenceImageWidth = imgData.Width;
        Config.InferenceImageHeight = imgData.Height;
        
        if (preprocessingMode == 0)
        {
            imgData = YoloHelper.PictureZoom(imgData,Config);
            Config.InputTensor = YoloHelper.PicWriTensor_Memory(imgData, Config.InputTensorInfo);
        }
        if (preprocessingMode == 1)
        {
            Config.InputTensor = YoloHelper.NoInterpolationWriteTensor(imgData,Config);
        }
        
        //容器
        IReadOnlyCollection<NamedOnnxValue> container = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(Config.ModelInputName, Config.InputTensor)
        };
        
        // 过滤数据数组
        List<YoloData> result = new List<YoloData>();
        
        // 获取任务模式处理策略
        var taskModelStrategy = TaskModelInferenceStrategyFactory.CreateTaskModelStrategy(Config.ExecutionTaskMode,Config.YoloVersion);
        
        // 策略处理
        result = taskModelStrategy.ExecuteTask(Config,container,confidenceDegree,iouThreshold,allIou);

        YoloHelper.RestoreCoordinates(result,Config);
        
        if (Config.ExecutionTaskMode != 0)
        {
            YoloHelper.RemoveCoordinates(result,Config);
        }

        return result;
    }
    
    
}