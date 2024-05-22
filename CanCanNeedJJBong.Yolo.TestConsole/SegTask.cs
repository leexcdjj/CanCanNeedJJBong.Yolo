namespace CanCanNeedJJBong.Yolo.TestConsole;

class SegTask : YoloTask
{
    protected override (string testImgName, string modelName, string action, int taskMode) GetTaskDetails()
    {
        return ("83c6a5691f1464f9c0e963f5d42bf3e0.jpeg", "yolov8n-seg.transd.onnx", "Seg", 3);
    }
}