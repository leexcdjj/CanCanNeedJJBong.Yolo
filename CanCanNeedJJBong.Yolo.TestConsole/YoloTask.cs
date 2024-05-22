using System.Drawing;
using CanCanNeedJJBong.Yolo.Core;

namespace CanCanNeedJJBong.Yolo.TestConsole;

abstract class YoloTask
{
    protected const float ConfidenceDegree = 0.5f;
    protected const bool AllIou = false;
    protected const float IouThreshold = 0.3f;

    protected static readonly string CurrentDirectory = Directory.GetCurrentDirectory();
    protected static readonly string ModelsDirectory = Path.Combine(CurrentDirectory, "Models");
    protected static readonly string TestImgDirectory = Path.Combine(CurrentDirectory, "TestImg");
    protected static readonly string GenImgDirectory = Path.Combine(CurrentDirectory, "GenImg");

    public void Execute()
    {
        EnsureDirectoryExists(GenImgDirectory);

        var (testImgName, modelName, action, taskMode) = GetTaskDetails();
        string genImgName = $"{Path.GetFileNameWithoutExtension(testImgName)}{action}.jpg";
        string modelPath = Path.Combine(ModelsDirectory, modelName);
        string testImgPath = Path.Combine(TestImgDirectory, testImgName);
        string genImgPath = Path.Combine(GenImgDirectory, genImgName);

        var yolo = new YoloService(modelPath, false, 0, 8, taskMode);
        using var imgBit = new Bitmap(testImgPath);
        var data = yolo.ModelReasoning(imgBit, ConfidenceDegree, IouThreshold, AllIou, 1);
        using var img = yolo.GenerateIma(imgBit, data);

        img.Save(genImgPath);

        Console.WriteLine($"{action} 推理生成完成");
    }

    protected abstract (string testImgName, string modelName, string action, int taskMode) GetTaskDetails();

    private static void EnsureDirectoryExists(string directory)
    {
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }
}