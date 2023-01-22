Follow me to reproduce the results of single wheel and full VIPER rover.

Step1: Clone a brand new Chrono and check out to this commit: 75a35c3bf7d76a0408c7a69400db5605c0347a2b

Step2: Replace all the files you see in /src of Chrono with the files you see in this folder. If you don't find a file in /src but you see one here, add them to the FSI demo directory and modify the CmakeList.txt accordingly.

Step3: Build Chrono

Step4：Run the demo you would like to run, e.g the demo_FSI_SingleWheelTest.cpp

PS: 
1. For the single wheel test on real slope terrain ,run this demo demo_FSI_SingleWheelTest_realSlope.cpp
2. Change density, friction coefficient, gravity, and wheel mass in the demo (not in the JSON file)
3. Change the damping coefficient and Young's modulus in the JSON file (not in the demo)







