# movie_generative_transformer

## Overview 

Generates random movie plot with transformer decoder architecture using PyTorch.

## Current Progress

Currently modifying hyperparameters to minimize loss. Implemented dropout with significant improvements.
The project only includes the decoder portion of the transformer, plans for the project would include the implementation of the encoder.

## Create pipenv and install dependencies 

Install Python 3.10.12

```shell 
cd ./path/to/project
pip install pipenv --user
pipenv install
```
## Running and Training

To train network:

```shell
pipenv shell
cd src
python moviegen.py
```
Models and training logs will be stored in ``` /src/out ``` folder.

## Results

### First train:

Trained for 8000 iterations. No dropout and no layer norms between attention blocks.

![Loss Graph](https://github.com/ethancpwoo/movie_gen_transformer/blob/main/src/out/losschart-8000iters.png?raw=true)

Result: 

ysapualdabneunr.re it eo  oh. a co  seqipnvge ael.e sttla hfarehu hmciCJnn.dtr eelpoebhtexrghhwatlrea tedfwMhgs gii mo fb syp   eeh aebe   yelesifns ivufwaf - evetseb  tyotsrmvsPiepLsscnsnaasi ahdcthg

### Second training:

Trained for 4500 iterations since graph from first training showed minimum loss at ~4500 iteration. Could this be because of overfitting? Overcomplicated network for such a simple dataset? Aggressive learning rate? Further testing is needed but training takes ~30 mins to 45 mins.

![Loss Graph](https://github.com/ethancpwoo/movie_gen_transformer/blob/main/src/out/losschart.png?raw=true)

Result:

Genre: action/screen 

Description:
Pivil with Meiniciet, who tries to learn the house of the dirty himself of his job. Riv telves them to do, however, so as a pole onto Susie's hooffared to Alice. Agner arrives to Homes. Meanwhile, who has been killed by Jerice and gives home fights. Om wants to his griting, Al Ate becomes withdring the Confederal Computtany country rusiness. Grandman overs his way. Wilderward Grece to move with Byan's boyfriend, Obsesva becomes a job. He does the training as a leader after rescuing him for a more with Gyramjee's physical ceal company. When they go out any may each other first-fletee, Lonovelyn and Fonder she discovers or the Nazi, but also rides it back in a mafirs.
A sked, and osball a pact at Ishan and smashes him in tramplains. She is climb in an attack after her son Tim's hand, but would be quickles willing to allud how they under that she sends Siron. After a frustration, Jeha, the womanizers and manages to a honeymoon fatallinal. While posing r

### Third training: 

Trained for 8000 iterations with dropout and layer norms between attention heads. From dropout paper, it should prevent overfitting massively and allow for more training iterations thus back to 8000 iters. 

![Loss Graph](https://github.com/ethancpwoo/movie_gen_transformer/blob/main/src/out/losschartdropout8000iter.png?raw=true)

Result

Title: Eventually Broads of a and damaged

Genre: feedy

Description:, a believers is that Lady is her dreaming in the training tables off is the back. Near of Sylvie's favourite guards, Mr. Mahaiukhi) and brushing with Amaton and Mary are caught in by the guards of agree on Paul, a man of aeroic letter of helping Bosha (Towensen) and Paul (Aaron Hartrick). Lau is a then to antidote the guards to cheach on the countrying. He meets and contains a mystery to have get married light to light like out light yes against Ralphton.    
At Shrikopalani retrieves it from Janagayakrishna (Sandha Dr), finding his lonely and taken crossing the guards in the course of his future machines, information on the locals. He is unable to single themselves they all Wallham to a vision who lives in Norbide's business path. She plans to rapes Patil and May at his left, who and he wake up to the doll. His father's deep attack in a suit accident. The nowless accompanie of the head ranks, quitting the camera and own
