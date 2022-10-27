This example contains a 7 DOF arm attempting to keep its end effector 0.5m above the ground. Using a PD controller, this requirement is breached 

![7DOFNOCBFValue](https://user-images.githubusercontent.com/32720154/198381750-5ad10fa1-c864-4d76-be0d-884a8929ee7a.png)


| Frame 0  | Frame 56 | Frame 300 |
| ------------- | ------------- | ------------- |
| ![Frame0](https://user-images.githubusercontent.com/32720154/198381794-fb84ff38-47e3-49ee-b396-4956476712b6.png)  |![Frame56NOCBF](https://user-images.githubusercontent.com/32720154/198381803-852b0993-2088-4e08-aaef-7c381f32dcbf.png) | ![Frame300NOCBF](https://user-images.githubusercontent.com/32720154/198381804-e23872d1-1e03-4208-93fa-33254d392c89.png)|

Adding a CBF representing the height of the end effector - 0.5m, the requirement is upheld 


![7DOFCBFValue](https://user-images.githubusercontent.com/32720154/198381905-afdaf067-b15a-4c69-9fd1-e26bb96fac72.png)

| Frame 0  | Frame 56 | Frame 300 |
| ------------- | ------------- | ------------- |
| ![Frame0](https://user-images.githubusercontent.com/32720154/198381908-65c69033-f140-4128-8f0a-16cf01e4c0b7.png)| ![Frame56CBF](https://user-images.githubusercontent.com/32720154/198381911-0dcb10fd-7cea-43c9-be70-080dfdd42935.png) | ![Frame300CBF](https://user-images.githubusercontent.com/32720154/198381919-deeae564-8d3f-4af7-bd17-e4d4aad11a2e.png) |
