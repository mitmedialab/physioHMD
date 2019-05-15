# PhysioHMD

<img src="https://github.com/mitmedialab/physioHMD/blob/master/Figures/Artboard.png" width="100%" title="hover text">

The  PhysioHMD platform introduces a software and hardware modular interface built for collecting affect and physiological data from users wearing a head-mounted display. The platform enables researchers and developers to aggregate and interpret signals in real-time and use them to develop novel, personalized interactions, as well as evaluate virtual experiences. Our design offers seamless integration with standard HMDs, requiring minimal setup effort for developers and those with less experience using game engines. The PhysioHMD platform is a flexible architecture that offers an interface that is not only easy to extend but also complemented by a suite of tools for testing and analysis. We hope that PhysioHMD can become a universal, publicly available testbed for VR and AR researchers.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

<img src="https://github.com/mitmedialab/physioHMD/blob/master/Figures/facePlates_03.jpg" width="100%" title="hover text">

The saliva acquisition and sensing module consists of a mechanical arrangement for acquiring a small quantity of saliva from \textit{buccal mucosa} (inside of the cheek). It consists of two stepper motors, a small PTFE tube, a paper sensor cartridge and the electronics for digital sensing. The PTFE tube of diameter 2mm is held against the inside part of the cheek while the rest of the device sits outside the cheek. A stepper motor is used to actuate the tube linearly back and forth to gather a small amount of saliva. On the tube's way out the gathered saliva is transferred onto the paper sensor by contact. Another stepper motor is then used to advance the paper sensor cartridge. A new portion of the paper sensor is rolled out and swabs the saliva from the tube.
This motion is repeated several times to get the appropriate amount of saliva onto the paper sensor. The synchronous motion of the paper sensor and the tube allows the saliva to be transferred from the tube to the sensor without contaminating an unused sensor. 

A color RGB sensor is mounted orthogonal to the paper sensor to track the color changes. We used the TCS34725 RGB color sensor with white LED light for illumination. The system is controlled by a BLE enabled microcontroller(BC832) mounted on a PCB inside the casing. The system also has a 9-axis Inertial Measurement Unit (MPU9250) to track orientation as well as actions of the individual. The system is powered by a single cell LiPo battery of capacity 100mAH.


### Materials


<img src="https://github.com/mitmedialab/physioHMD/blob/master/Figures/physioHMD_exploded.jpg" width="100%" title="hover text">


## Contributors
V.0
* Guillermo Bernal
* Abhinandan Jain
* Tao Yang
* Pattie Maes

## License

This project is an opensource project

## Acknowledgments

* MIT Media Lab
