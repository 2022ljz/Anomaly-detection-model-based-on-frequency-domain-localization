## 指南
### 1.请在`python3.9`环境中使用项目的`requirements.txt`进行环境配置：
`pip install -r requirements.txt`

### 2.请安装`gpu-pytorch == 2.5.0`（可以直接去官网安装）：
`pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121`

示例CUDA=12.1，请根据自己设备的CUDA版本安装。

### 3.通过如下命令行运行脚本进行实验：
`$ sh scripts/MSL.sh `

`$ sh scripts/SMAP.sh `

## 异常检测性能
注意：本项目已经在`checkpoints`文件夹下提前训练好了一组模型，想要直接看异常检测效果，请在`scripts`文件夹下将对应的脚本文件中训练部分（第一行）的`num_epochs`参数设置为`0`，再去运行脚本文件。这样，会直接调用训练好的模型在测试集上进行异常检测并输出结果。
<table>
  <tr>
    <th colspan="4" style="text-align: center">MSL</th>
    <th colspan="4" style="text-align: center">SMAP</th>
  </tr>
  <tr>
    <th style="text-align: center">Accuracy</th>
    <th style="text-align: center">Precision</th>
    <th style="text-align: center">Recall</th>
    <th style="text-align: center">F1-score</th>
    <th style="text-align: center">Accuracy</th>
    <th style="text-align: center">Precision</th>
    <th style="text-align: center">Recall</th>
    <th style="text-align: center">F1-score</th>
  </tr>
  <tr>
    <th style="text-align: center">98.96</th>
    <th style="text-align: center">92.84</th>
    <th style="text-align: center">97.70</th>
    <th style="text-align: center">95.21</th>
    <th style="text-align: center">99.08</th>
    <th style="text-align: center">93.96</th>
    <th style="text-align: center">99.15</th>
    <th style="text-align: center">96.48</th>
  </tr>
</table>

		
