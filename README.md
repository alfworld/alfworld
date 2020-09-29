# TW-Alfred

Code for anonymous ICLR 2021 submission *BUTLER: Building Understanding in Textworld via Language for Embodied Reasoning*

This code base is developed based on the [official Alfred codebase](https://github.com/askforalfred/alfred).



### Dependencies

We will provide docker images to ensure the reproducibility of our work. 
During the review period, we follow the anonymous guidance and thus hide this information.
This work requires dependencies include the libraries listed in "requirements.txt" and our customized version of TextWorld and Fast Downward, which are developed based on [TextWorld](https://github.com/microsoft/TextWorld) and [Fast Downward](https://github.com/aibasel/downward).


### Training Text agents

```
# Download data
cd data
wget https://bit.ly/3cIEx9R
wget https://bit.ly/3mZsrhf
wget https://bit.ly/2S8lexl 
unzip 'json_2.1.1_*.zip' ; cd ../

# Set ALFRED_ROOT
export ALFRED_ROOT=$$(pwd)
cd models/dqn/

# Train DAgger
python train_dagger.py config.yaml
```




    
