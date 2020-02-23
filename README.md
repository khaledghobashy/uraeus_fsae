# **URAEUS** | FSAE

A multi-body systems database for **FSAE** vehicles implemented in the **uraeus** open-source framework.

----------------

## Description

This repository aims to provide:

- A full-featured database of multi-body system models commonly used in developing **Formula Student** vehicles.
- A showcase of the **uraeus** open-source framework capabilities, and a "How To" use it in real-world modelling situations.
- A full-featured modelling and simulation process, i.e. model creation, numerical simulation and 3D visualization.

### URAEUS

*Brief about the uraeus framework*

### Features

Currently, the database provides:

#### Symbolic Models
These are various template-based symbolic topologies that represent different multi-body mechanisms that can be found in a typical **Formula Student** vehicle. 
This is a list of the currently modelled topologies:

- **Suspension Mechanisms**:
  - Double Wishbone Direct-Acting mechanism
  - Double Wishbone Bellcrank-Actuated mechanism
- **Steering Mechanisms**:
  - Steering Rack
  - Parallel Link Steering
- **Vehicle Chassis**
  - Rigid Chassis



#### Symbolic Assemblies

These are symbolic topologies that assemble various template-based models together, constructing a bigger multi-body system, such as full-vehicle assemblies and suspension test-rigs.

- **Full-vehicle Assembly**
  A symbolic assembly representing a full-vehicle with the following subsystems:
  - **Front Axle**
    Double Wishbone Bellcrank-Actuated mechanism
  - **Rear Axle**
    Double Wishbone Bellcrank-Actuated mechanism
  - **Front Steering**
    Steering Rack
  - **Vehicle Chassis**
    Rigid Chassis

