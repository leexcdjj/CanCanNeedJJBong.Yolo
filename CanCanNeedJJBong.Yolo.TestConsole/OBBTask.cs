namespace CanCanNeedJJBong.Yolo.TestConsole;

class OBBTask : YoloTask
{
    protected override (string testImgName, string modelName, string action, int taskMode) GetTaskDetails()
    {
        return ("OBB测试球场.jpg", "yolov8n-obb.transd.onnx", "OBB", 6);
    }
}