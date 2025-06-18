# Alex Greenacre KF7029 - Dissertation 
This project covers the research that I carried out as part of my masters dissertation during my Msc Advanced computer science course.
<br>
The topic of the dissertation covers the radar based localization of humans using through wall radar and image transformers, using a dataset created by Schroth,et al(2023) models CNN,ResNet30 and a image transformer novel to the field were trained and compared against to evaluate the image transformers ability in this field.     
<br>
This project contains:
<ul>
<li>Project code</li>
<li>Associated dissertation discussing results</li>
</ul>

## References 
Dataset used in this paper 
<br>
C. A. Schroth et al., Emergency Response Person Localization and Vital Sign Estimation Using a
Semi-Autonomous Robot Mounted SFCW Radar. 2023. https://arxiv.org/abs/2305.15795
<br>
Paper related to the dataset 
<br>
C. A. Schroth, C. Eckrich, S. Fabian, O. von Stryk, A. M. Zoubir, and M. Muma, Multi-Person
Localization and Vital Sign Estimation Radar Dataset. IEEE Dataport, 2023. doi: https://doi.org/10.21227/4bzd-jm32.

## How to use 
### Training Models
Due to github repo size weights have been removed, to retrain the models run the training scenarios using base command bellow although this will lead to different results than that in the test results folder  

### Running Tests
All tests are ran from the <b>Scenarios</b> folder to run a test use the following code, all tests output either to command line or as charts in the <b>Plots</b> folder  
Base Command:<br>
<code>
python3 -m Scenarios.Folder_name.File_name 
</code>


Example:<br> 
<code>
python3 -m Scenarios.Scenario_4.CombinedObjectTest
</code>  

#### Note 
ResNet model will require a internet connection when first ran to download the reuqired data from the keras_cv libary 

### File Structure
<ul>
Alex Greenacre Dissertation(project folder) 
<ul>
<li>
Data
</li> 
<li>Models</li>
<li>OldFiles_PerliminaryTests</li>
<li>Plots</li>
<li>Scenarios</li>
<li>Test_Results</li>
<li>Utils</li>
</ul>
ViT_Dissertation___Alex_Greenacre(dissertation pdf)
</ul>


#### Data 
Contains all the radar data and labels, dataset can be found in the reference section at the top of the paper
 
#### Models 
Contains the classes for all models used in the experiments 
models include CNN,ResNet,ViT with the lstm_cnn model used in a failed ECG test 

Custom layers are also stored in this folder 

As well as the models the .h5 weight files are stored in this folder to allow for each model to be loaded for reliability in collecting results and without having to re train it for every scenario
#### Plots 
Contains the output plots for each scenario, saves the plots into the Test_Result folder  
#### Scenarios 
Stores the files used to run the Experiments and training of the models 

#### Test Results 
Any output which has not been displayed on the command line will be displayed here 

#### Utils 
Stores files essential to model and tests<br> 
includes loading of data, and what each file represents. Also contains the custom loss function used in the models evaluation  


        

