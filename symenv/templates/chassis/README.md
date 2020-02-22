# RIGID CHASSIS

**TEMPLATE-BASED TOPOLOGY**

------------------------------------------------

### **Summary**

This Chassis template topology represents a simple topology of one rigid body with one generic force representing the aerodynamic drag force acting on the vehicle body and no joint constraints. 

--------------------------------------

### **Topology Layout**

The mechanism consists of 1 Body only. Therefore, total system coordinates is $n=n_b\times7 = 1\times7 = 7$, where $n_b$ is the total number of bodies. [^1]

The list of bodies is given below:

- Chassis

The system has no algebraic constraints as there is no joints defined in this template.

</br>

------------------------------------------------------
<br/>

[^1]: The tool uses [euler-parameters](https://en.wikibooks.org/wiki/Multibody_Mechanics/Euler_Parameters) -which is a 4D unit quaternion- to represents bodies orientation in space. This makes the generalized coordinates used to fully define a body in space to be **7,** instead of **6**, it also adds an algebraic equation to the constraints that ensures the unity/normalization of the body quaternion. This is an important remark as the calculations of the degrees-of-freedom depends on it.

