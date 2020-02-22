# Parallel Link Steering

**TEMPLATE-BASED TOPOLOGY**

------------------------------------------------

### **Summary**

The parallel link steering mechanism is a simple planar four-bar linkage consisting two rockers -left and right- connected to the vehicle chassis from one end, and connected to a coupler part from the other end, forming a closed kinematic loop.

In the full vehicle assembly, these rockers are connected to the tie-rods of the suspension mechanism to control the steering degree-of-freedom of the wheel-assembly in the connected suspension subsystem.

![Figure 1 - System Layout](figure.png)

*Figure 1 - System Layout*

--------------------------------------

### **Topology Layout**

The mechanism consists of 3 actual bodies and 1 virtual body representing the vehicle chassis body. Therefore, total system coordinates -excluding the chassis- is $ n=n_b\times7 = 3\times7 = 21 $, where $n_b$ is the total number of bodies. [^1]

The list of actual bodies is given below:

- Coupler
- Right Rocker
- Left Rocker

The system connectivity is given in the table below.

<center>

| Joint Name             | Body i       | Body j  | Joint Type | $n_c$ |
| :--------------------- | :----------- | :------ | :--------: | ----: |
| Right Rocker - Coupler | Right Rocker | Coupler | Spherical  |     3 |
| Left Rocker - Coupler  | Left Rocker  | Coupler | Universal  |     4 |
| Right Rocker - Chassis | Right Rocker | Chassis |  Revolute  |     5 |
| Left Rocker - Chassis  | Left Rocker  | Chassis |  Revolute  |     5 |
| **Total**              |              |         |            |    17 |

</center>

</br>

Hence, the total number of constraints equations is:
$$ n_{c} = n_{c_j} + n_{c_p} + n_{c_g} = 17 + (3\times 1) + 0 = 20 $$

where:
* $n_{c_j}$ is the joints constraints.
* $n_{c_p}$ is the euler-parameters normalization constraints.
* $n_{c_g}$ is the ground constraints.

Therefore, the resulting **DOF** is:
$$ n - n_c = 21 - 20 = 1 $$

which is the expected one DOF of the four-bar linkage.

------------------------------------------------------
<br/>

[^1]: The tool uses [euler-parameters](https://en.wikibooks.org/wiki/Multibody_Mechanics/Euler_Parameters) -which is a 4D unit quaternion- to represents bodies orientation in space. This makes the generalized coordinates used to fully define a body in space to be **7,** instead of **6**, it also adds an algebraic equation to the constraints that ensures the unity/normalization of the body quaternion. This is an important remark as the calculations of the degrees-of-freedom depends on it.

