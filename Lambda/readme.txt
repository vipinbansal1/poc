throatimageModel_Running_onGPU_0.py
This code pure tensorflow code, where we enforced to run on GPU 0
config.gpu_options.visible_device_list = str("0")


throatimageModel_onGPU1_With_horovod.py
This code using Horovord and specifically running on GPU1 only.
os.environ["CUDA_VISIBLE_DEVICES"]="1"   #this could be the reason but not sure.


throatimageModel_WithHorovod.py
Horovod code with multiple gpu

horovodex1.py, horovodTest.py   #open code

mnist_Sequential_Withhorovod.py
Horovod with sequential API code.


mnistmirrorsgtrtegy.py
Mirror straetgy working on sequential API.

ms_throatModel.py

MS using session() api's not working . Memory footprint says both GPU's utlized.

---------------------------------: 35.18801164627075 mirror startegy
27 when gpu
horovod 
[1,0]<stdout>:---------------------------------: 48.181734561920166


throoat image
ms,...not sure if code is right....consuming ythe mem of both gpu but gpu1 utilization is 0 Time ****************************** 39.29556202888489



withhorovod on signle gpu   [1,0]<stdout>:Time ****************************** 41.22106313705444

with gpu1 only [1,0]<stdout>:Time ****************************** 39.77163505554199
horovod 2 gpu [1,0]<stdout>:Time ****************************** 63.27406930923462




[0] GeForce RTX 2080 Ti | 43'C,  13 % |   849 / 11019 MB | vipin(841M) gdm(4M)
[1] GeForce RTX 2080 Ti | 40'C,  14 % |   890 / 11014 MB | vipin(841M) gdm(36M) gdm(8M)


[0] GeForce RTX 2080 Ti | 39'C,   6 % | 10829 / 11019 MB | vipin(10821M) gdm(4M)
[1] GeForce RTX 2080 Ti | 40'C,   7 % | 10826 / 11014 MB | vipin(10777M) gdm(36M) gdm(8M)

how we can combine these two gpus to form one gpu?
NVLink

watch -n 1 nvidia-smi



