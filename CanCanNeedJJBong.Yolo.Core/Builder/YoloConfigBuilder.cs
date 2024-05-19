using Microsoft.ML.OnnxRuntime;

namespace CanCanNeedJJBong.Yolo.Core.Builder;

public class YoloConfigBuilder
{
    private readonly YoloConfig _yolo;

    public YoloConfigBuilder()
    {
        _yolo = new YoloConfig();
    }
    
    /// <summary>
    /// 设置模型会话，包括模型路径和是否启用GPU。
    /// </summary>
    /// <param name="modelPath">模型文件路径</param>
    /// <param name="enableGpu">是否启用GPU</param>
    /// <param name="gpuIndex">GPU索引</param>
    /// <returns>Builder对象，用于链式调用</returns>
    public YoloConfigBuilder SetModelSession(string modelPath, bool enableGpu = false, int gpuIndex = 0)
    {
        if (enableGpu)
        {
            var options = new SessionOptions();
            options.AppendExecutionProvider_DML(gpuIndex);
            _yolo.ModelSession = new InferenceSession(modelPath, options);
        }
        else
        {
            _yolo.ModelSession = new InferenceSession(modelPath);
        }

        _yolo.ModelInputName = _yolo.ModelSession.InputNames.First();
        _yolo.ModelOutputName = _yolo.ModelSession.OutputNames.First();
        _yolo.InputTensorInfo = _yolo.ModelSession.InputMetadata[_yolo.ModelInputName].Dimensions;
        _yolo.OutputTensorInfo = _yolo.ModelSession.OutputMetadata[_yolo.ModelOutputName].Dimensions;

        return this;
    }
    
    /// <summary>
    /// 设置模型的元数据，包括标签组、模型版本和任务类型。
    /// </summary>
    /// <returns>Builder对象，用于链式调用</returns>
    public YoloConfigBuilder SetModelMetadata()
    {
        var modelMetadata = _yolo.ModelSession.ModelMetadata.CustomMetadataMap;

        if (modelMetadata.ContainsKey("names"))
        {
            _yolo.LabelGroup = SplitLabelSignature(modelMetadata["names"]);
        }
        else
        {
            _yolo.LabelGroup = new string[0];
        }

        if (modelMetadata.ContainsKey("version"))
        {
            _yolo.ModelVersion = modelMetadata["version"];
        }

        if (modelMetadata.ContainsKey("task"))
        {
            _yolo.TaskType = modelMetadata["task"];
            if (_yolo.TaskType == "segment")
            {
                string outputName2 = _yolo.ModelSession.OutputNames[1];
                _yolo.OutputTensorInfo2_Segmentation = _yolo.ModelSession.OutputMetadata[outputName2].Dimensions;
                _yolo.SemanticSegmentationWidth = _yolo.OutputTensorInfo2_Segmentation[1];
                _yolo.MaskScaleRatioW = 1f * _yolo.OutputTensorInfo2_Segmentation[3] / _yolo.InputTensorInfo[3];
                _yolo.MaskScaleRatioH = 1f * _yolo.OutputTensorInfo2_Segmentation[2] / _yolo.InputTensorInfo[2];
            }
            else if (_yolo.TaskType == "pose")
            {
                if (_yolo.OutputTensorInfo[1] > _yolo.OutputTensorInfo[2])
                {
                    _yolo.ActionWidth = _yolo.OutputTensorInfo[2] - 5;
                }
                else
                {
                    _yolo.ActionWidth = _yolo.OutputTensorInfo[1] - 5;
                }
            }
        }
        else
        {
            if (_yolo.OutputTensorInfo.Length == 2)
            {
                _yolo.TaskType = "classify";
            }
            else if (_yolo.OutputTensorInfo.Length == 3)
            {
                if (_yolo.ModelSession.OutputNames.Count == 1)
                {
                    _yolo.TaskType = "detect";
                }
                else if (_yolo.ModelSession.OutputNames.Count == 2)
                {
                    string outputName2 = _yolo.ModelSession.OutputNames[1];
                    _yolo.OutputTensorInfo2_Segmentation = _yolo.ModelSession.OutputMetadata[outputName2].Dimensions;
                    _yolo.SemanticSegmentationWidth = _yolo.OutputTensorInfo2_Segmentation[1];
                    _yolo.MaskScaleRatioW = 1f * _yolo.OutputTensorInfo2_Segmentation[3] / _yolo.InputTensorInfo[3];
                    _yolo.MaskScaleRatioH = 1f * _yolo.OutputTensorInfo2_Segmentation[2] / _yolo.InputTensorInfo[2];
                    _yolo.TaskType = "segment";
                }
            }
            else
            {
                throw new Exception("Unsupported model type");
            }
        }
        
        // 设置张量宽度和张量高度
        _yolo.TensorWidth = _yolo.InputTensorInfo[3];
        _yolo.TensorHeight = _yolo.InputTensorInfo[2];
        return this;
    }
    
    /// <summary>
    /// 设置推理参数，包括YOLO版本、张量宽度和张量高度。
    /// </summary>
    /// <param name="yoloVersion">YOLO版本</param>
    /// <param name="tensorWidth">张量宽度</param>
    /// <param name="tensorHeight">张量高度</param>
    /// <returns>Builder对象，用于链式调用</returns>
    public YoloConfigBuilder SetInferenceParameters(int yoloVersion,int taskMode)
    {
        _yolo.TaskMode = taskMode;
        _yolo.YoloVersion = GetModelVersion(yoloVersion);
        return this;
    }
    
    /// <summary>
    /// 分割标签名。
    /// </summary>
    /// <param name="name">标签名字符串</param>
    /// <returns>分割后的标签名数组</returns>
    private string[] SplitLabelSignature(string name)
    {
        // 删除括号
        name = name.Replace("{", "").Replace("}", "");
        
        // 分割数组
        string[] SplitArr = name.Split(',');
        
        string[] result = new string[SplitArr.Length];
        
        for (int i = 0; i < SplitArr.Length; i++)
        {
            int start = SplitArr[i].IndexOf(':') + 3;
            int end = SplitArr[i].Length - 1;
            result[i] = SplitArr[i].Substring(start, end - start);
        }

        return result;
    }

    public YoloConfig GetYoloConfig()
    {
        return _yolo;
    }
    
    /// <summary>
    /// 获取模型版本
    /// </summary>
    /// <param name="version"></param>
    /// <returns></returns>
    private int GetModelVersion(int version)
    {
        var taskType = _yolo.TaskType;
        var modelVersion = _yolo.ModelVersion;
        var outputTensorInfo = _yolo.OutputTensorInfo;
        var labelGroup = _yolo.LabelGroup;
        var semanticSegmentationWidth = _yolo.SemanticSegmentationWidth;
        
        if (taskType == "classify")
        {
            return 5;
        }

        if (version >= 8)
        {
            return 8;
        }
        else if (version < 8 && version >= 5)
        {
            return version;
        }

        if (modelVersion != "")
        {
            return int.Parse(modelVersion.Split('.')[0]);
        }

        int mid = outputTensorInfo[1];
        int right = outputTensorInfo[2];
        int size = mid < right ? mid : right;
        
        // 标签数量
        int lableCount = labelGroup.Length;
        if (lableCount == size - 4 - semanticSegmentationWidth)
        {
            return 8;
        }

        if (lableCount == 0 && mid < right)
        {
            return 8;
        }

        return 5;
    }
    
}