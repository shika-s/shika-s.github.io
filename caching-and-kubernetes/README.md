[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VsEYUSpQ)
# Lab 3: Caching and Kubernetes

<p align="center">
    <!--FAST API-->
        <img src="https://user-images.githubusercontent.com/1393562/190876570-16dff98d-ccea-4a57-86ef-a161539074d6.svg" width=10% alt="FastAPI">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=10% alt="Plus">
    <!--REDIS LOGO-->
        <img src="https://user-images.githubusercontent.com/1393562/190876644-501591b7-809b-469f-b039-bb1a287ed36f.svg" width=10% alt="Redis">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=10% alt="Plus">
    <!--KUBERNETES-->
        <img src="https://user-images.githubusercontent.com/1393562/190876683-9c9d4f44-b9b2-46f0-a631-308e5a079847.svg" width=10% alt="Kubernetes">
</p>

- [Lab 3: Caching and Kubernetes](#lab-3-caching-and-kubernetes)
  - [Lab Overview](#lab-overview)
  - [How to approach this lab](#how-to-approach-this-lab)
  - [Lab Objectives](#lab-objectives)
    - [API Requirements](#api-requirements)
    - [Deployment Requirements](#deployment-requirements)
      - [Understanding Kubernetes YAML Structure](#understanding-kubernetes-yaml-structure)
      - [Example: namespace.yaml](#example-namespaceyaml)
    - [Documentation Requirements](#documentation-requirements)
    - [Lab Setup](#lab-setup)
    - [Expected Final Folder Structure](#expected-final-folder-structure)
    - [Grading Script](#grading-script)
  - [Helpful Tips](#helpful-tips)
    - [API Diagram](#api-diagram)
    - [Deployment Diagram](#deployment-diagram)
    - [Redis Expectations](#redis-expectations)
      - [Dependency Adding](#dependency-adding)
      - [How to use `fastapi-simple-redis-cache`](#how-to-use-fastapi-simple-redis-cache)
      - [Running Redis Locally](#running-redis-locally)
      - [Running Redis in Kubernetes](#running-redis-in-kubernetes)
      - [Making Code Reusable When Developing](#making-code-reusable-when-developing)
      - [Verifying your Redis deployment](#verifying-your-redis-deployment)
      - [Redis Deployment](#redis-deployment)
    - [Input Vectorization](#input-vectorization)
      - [Vectorization Conceptual Example](#vectorization-conceptual-example)
    - [Pytest, FastAPI, and Lifespan Events](#pytest-fastapi-and-lifespan-events)
    - [Kubectl and minikube](#kubectl-and-minikube)
    - [Networking Requirements](#networking-requirements)
    - [Docker Daemons](#docker-daemons)
    - [Specialized Containers: Init + Readiness + Liveness](#specialized-containers-init--readiness--liveness)
      - [Init Containers](#init-containers)
      - [Startup Probe](#startup-probe)
      - [Readiness Probe](#readiness-probe)
      - [Liveness Probe](#liveness-probe)
    - [Troubleshooting Common Issues](#troubleshooting-common-issues)
    - [Debugging Commands](#debugging-commands)
  - [Grading](#grading)
  - [Time Expectations](#time-expectations)

## Lab Overview

The goal of `lab3` is to extend your `FastAPI` application from `lab2` with the following:

- Your API takes a *list* of inputs to predict instead of a single input.
- Your API will have a rudimentary Redis cache for the new `/bulk-predict` endpoint based on the inputs.
- Your application deploys locally on a Kubernetes environment (Minikube).

## How to approach this lab

We have done our best to organize this lab top-down to direct your order of operations best. Focus on doing all the [API Requirements](#api-requirements) first, then work on [Deployment Requirements](#deployment-requirements). Start early, read all the hints we have given, and ask questions on slack.

> [!IMPORTANT]
> The Automatic feedback system via github actions is a shared resource across the class. The system runs smoothest with on-time submissions. Later submissions can experience delays due to increased load on the automated feedback system.

---

## Lab Objectives

### API Requirements

- [x] All requirements from `lab2`
- [ ] Ensure the model is loaded only when the container is instantiated/started (i.e., not every time you run a prediction)
- [ ] Create a new request model which extends your single input model to accept a list of inputs instead of a single input.
  - Use `houses` as the field name which expects a list of the input objects you designed in lab2
- [ ] Create a response model returning a `list` of floats
- [ ] Create a new `POST` endpoint `/bulk-predict` which takes a `List` of inputs based on the request model you created above
  - You must utilize the `multi_predict` function we have defined for you in the template
  - Add the decorator key to this defined function
  - [ ] Input

      ```{JSON}
      {
        "houses": [
          {
            "MedInc": 1,
            "HouseAge": 1,
            "AveRooms": 3,
            "AveBedrms": 3,
            "Population": 3,
            "AveOccup": 5,
            "Latitude": 1,
            "Longitude": 1
          },
          {
            "MedInc": 0,
            "HouseAge": 1,
            "AveRooms": 3,
            "AveBedrms": 3,
            "Population": 3,
            "AveOccup": 5,
            "Latitude": 1,
            "Longitude": 1
          }
        ]
      }
      ```

    - [ ] Output

      ```{JSON}
      {
        "predictions": [
          123.45,
          123.45
        ]
      }
      ```

- [ ] Run your predictions on a matrix input instead of row by row. (See [Input Vectorization](#input-vectorization) for our expectations)
- [ ] Utilize the prefix of `w255-cache-prediction` for all Redis keys
- [ ] Add caching to **both** prediction endpoints:
  - [ ] Cache the entire input sent to `/predict` (your existing endpoint from lab2)
  - [ ] Cache the entire input sent to `/bulk-predict` (your new endpoint)
  - See [Redis Expectations](#redis-expectations) for implementation details
- [ ] Update your tests from `lab2` to give list inputs to your new endpoint

> [!Caution]
> DO NOT PROCEED UNTIL YOU HAVE FINISHED ALL API REQUIREMENTS

### Deployment Requirements

- [ ] Deploy your application to Kubernetes locally (Minikube)
  - [ ] (`namespace.yaml`) Deploy all components to a non-default `namespace` called `w255`
  - [ ] (`deployment-redis.yaml`) Deployment for Redis in `w255` namespace
    - See [Redis Expectations](#redis-expectations) for more details and requirements
  - [ ] (`deployment-pythonapi.yaml`) Deployment for your API in `w255` namespace
    - [ ] Your API deployment (`deplyoment-pythonapi.yaml`) should include an `initContainer`, `readinessProbe`, `livenessProbe`, and `startupProbe`
      - Init Containers:
        - [ ] Create an `initContainer` named `init-verify-redis-service-dns` should wait for the Redis DNS to become available
        - [ ] Create an `initContainer` named `init-verify-redis-ready` should wait for the Redis Service to become available
      - [ ] `readinessProbe` should wait for the API to be locally available by using the `/lab/health` endpoint
      - [ ] `livenessProbe` should monitor whether the API is responsive by using the `/lab/health` endpoint
      - [ ] `startupProbe` should wait for the API to be locally available by using the `/lab/health` endpoint
    - [ ] Your API deployment should have `3` replicas
  - [ ] (`service-redis.yaml`) Service for Redis in `w255` namespace
    - [ ] Your redis container should be using the base image `redis:7`
  - [ ] (`service-prediction.yaml`) Service for API in `w255` namespace
    - It should be a [LoadBalancer](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer) type

> [!NOTE]
> When we run the autograder, we will name your docker image `lab3`. This has implications on the YAML files you will be writing.

#### Understanding Kubernetes YAML Structure

All Kubernetes resources follow the same basic structure with four top-level fields:

```yaml
apiVersion: <api-group/version>  # Which Kubernetes API to use
kind: <ResourceType>             # What type of resource (Namespace, Deployment, Service, etc.)
metadata:                        # Information about the resource
  name: <resource-name>          # Required: unique name for this resource
  namespace: <namespace>         # Which namespace this belongs to (except for Namespace itself)
  labels:                        # Optional: key-value pairs for organizing/selecting resources
    app: myapp
spec:                            # Desired state - what you want this resource to do
  # Contents vary by resource type
```

| Field | Purpose | Example Values |
|-------|---------|----------------|
| `apiVersion` | API version for this resource type | `v1` (core), `apps/v1` (deployments) |
| `kind` | Type of resource | `Namespace`, `Deployment`, `Service` |
| `metadata.name` | Unique identifier | `w255`, `redis-deployment` |
| `metadata.namespace` | Which namespace to create in | `w255` (omit for Namespace resources) |
| `spec` | The desired configuration | Varies by resource type |

> [!TIP]
> Use `kubectl explain <resource>` to see documentation for any resource type. For example, `kubectl explain deployment.spec` shows all fields available in a Deployment's spec.

#### Example: namespace.yaml

Here is a complete example of the namespace configuration file to get you started:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: w255
```

This is the simplest Kubernetes resource you'll create (no `spec` needed). Apply it first with `kubectl apply -f infra/namespace.yaml` before applying other resources.

### Documentation Requirements

- [ ] Use the `README.md` file generated in the `lab3` folder to document how to build, run, and test code
- [ ] Answer several short answer questions (2-3 sentences) to help solidify your understanding of this lab. There is a prompt for the questions in `SHORT-ANSWERS.md`

### Lab Setup

Copy your code from lab 2 into a new lab folder `lab3`. You will use the same model you trained in `lab3`

Install `minikube` ([Docs](https://minikube.sigs.k8s.io/docs/start/))

Change Kubernetes version to match what we will deploy into Elastic Kubernetes Service (EKS) for future labs and projects, `1.34.3`.

`minikube start --kubernetes-version=v1.34.3`

### Expected Final Folder Structure

```{text}
.
├── .github
│   └── autogenerated_files_do_not_worry/
├── README.md
├── SHORT-ANSWER.md
└── lab3
    ├── Dockerfile
    ├── README.md
    ├── model_pipeline.pkl
    ├── poetry.lock
    ├── pyproject.toml
    ├── infra
    │   ├── deployment-pythonapi.yaml
    │   ├── deployment-redis.yaml
    │   ├── namespace.yaml
    │   ├── service-prediction.yaml
    │   └── service-redis.yaml
    ├── src
    │   ├── __init__.py
    │   ├── housing_predict.py
    │   └── main.py
    ├── tests
    │   ├── __init__.py
    │   ├── conftest.py
    │   └── test_src.py
    └── trainer
        ├── predict.py
        └── train.py
```

### Grading Script

Most of your work for this lab will be coordinating your Kubernetes deployment.

We will test your system in the following way; consider emulating the following steps to verify

1. Start up Minikube
1. Setup your docker daemon to build with Minikube (See [Docker Daemons](#docker-daemons))
1. Ensure your API model is trained. Run the training script if needed
1. Build the docker container of your application
1. Apply your k8s namespace
1. Apply your Deployments and Services
1. (if you choose to use `minikube tunnel` ) Create a separate `minikube tunnel` process and record the PID of that process **OR** `port-forward` a local port on your machine to your API service
1. Wait for your API to be accessible
   - This is a separate check from your init containers and other health checks.
   - This should run to make sure your API is accessible/ready from your run script
1. Validate all Lab 1/2 Endpoints
1. Validate all Lab 3 Endpoints
1. Validate the cache is properly recording values
1. Clean up after yourself
   - (if you used `minikube tunnel`) kill the `minikube tunnel` process you ran earlier
   - delete all resources in the `w255` namespace
   - delete the `w255` namespace
   - stop Minikube

> [!IMPORTANT]
> We provide the autograder as a form of automated feedback. After your submission, we will add additional tests to the autograder, which ensure the veracity of your solution; the requirements will not change but will ensure that you follow the requirements presented in this readme.
> For example, the autograder prior to submission deadline will not verify if your caching system is successful, but this will be tested for after the deadline

---

## Helpful Tips

### API Diagram

The following is a visualization of the sequence diagram describing our new API

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as API
    participant R as Redis
    participant M as Model

    U ->> A: POST JSON payload
    break Input payload does not satisfy pydantic schema
        A ->> U: Return 422 Error
    end
    A ->> R: Check if value is<br>already in cache
    alt Value exists in cache
        R ->> A: Return cached<br>value to app
    else Value not in cache
        A ->>+ M: Input values to model
        M ->>- A: Store returned values
        A ->> R: Store returned value<br>in cache
    end

    A ->> U: Return Values as<br>output data model
```

### Deployment Diagram

The following is a visualization of the infrastructure you should implement for Lab 3.

```mermaid
flowchart TB
User(User)
    subgraph k8s [Minikube Node]
    subgraph subgraph_padding1 [ ]
        subgraph cn [Common namespace: w255]
            direction TB
            subgraph subgraph_padding2 [ ]
            NPS2(LoadBalancer: prediction-service):::nodes
            subgraph PD [python-api-deployment]
                direction TB
                IC1(Init Container: init-verify-redis-service-dns)
                IC2(Init Container: init-verify-redis-ready)
                FA(Fast API Container):::fa

                IC1 --> IC2 --> FA
            end
            NPS1(ClusterIP: redis-service):::nodes
            RD(redis-deployment)

            NPS1 <-->|Port 6379| PD
            NPS1 <-->|Port 6379| RD
            NPS2 <-->|Port 8000| PD
        end
        end
    end
    end

User <---->|Minikube Tunnel or Port Forward <br/> Port:8000| NPS2

classDef nodes fill:#68A063
classDef subgraph_padding fill:none,stroke:none
classDef inits fill:#AF5FEE
classDef fa fill:#009485

style cn fill:#B6D0E2
style RD fill:#D82C20
style PD fill:#FFD43B
style k8s fill:#326ce5;

class subgraph_padding1,subgraph_padding2 subgraph_padding
class IC1,IC2 inits
```

### Redis Expectations

#### Dependency Adding

In this lab, we will utilize a naive approach to caching to protect you from repeated messages.
You are expected to use [`fastapi-simple-redis-cache`](https://pypi.org/project/fastapi-simple-redis-cache/) library that handles a lot of things for you.
Add version `^2.0.0` as a dependency to your poetry build.
Look into [dependency specification within poetry](https://python-poetry.org/docs/dependency-specification/) to make sure you add the exact correct version.

#### How to use `fastapi-simple-redis-cache`

There is [great documentation available on PyPi](https://pypi.org/project/fastapi-simple-redis-cache/) on how to use the library.
You will need to add the `NaiveCache` middleware to your application.
You will need to exclude your `/lab/health` endpoint so that its results will not be cached
We have added the cache prefix of `w255-cache-prediction` as part of the initialization template in this repo.
This means that all keys generated will have the associated prefix; **do not change this or you will fail autograder tests.**

#### Running Redis Locally

When working on the [API Requirements](#api-requirements), you will need to have some form of Redis available so that you can verify caching.
You can do this by running Redis with the following bash command.

```bash
docker run -d --name temp-redis -p 6379:6379 redis
```

You will then be able to access redis on `localhost` at port `6379`

When you run `Redis` as a docker container, there will be difficulties in connecting your FastAPI application if it is also being executed as a container.
This is a problem called "container orchestration" and we will use Kubernetes to more reliably deal with container to container networking.
We recommend finishing all API requirements locally before testing your application as a docker container.

```mermaid
flowchart TD
    subgraph doc [Docker Environment]
        dockerFastapi(FastAPI Container)
        dockerRedis(Redis Container)
    end

    subgraph loc [Local Execution]
        localFastapi(FastAPI Executed Locally)
    end

    localFastapi <== Easy ===> dockerRedis
    dockerFastapi <-. Difficult .-> dockerRedis

    linkStyle 0 stroke:#0a0,stroke-width:4px,color:green;
    linkStyle 1 stroke:#f00,stroke-width:4px,color:red;
```

#### Running Redis in Kubernetes

When running in kubernetes, your "local" network (i.e. `localhost`) will be inaccessible.

This means you will rely on your [kubernetes service](https://kubernetes.io/docs/concepts/services-networking/service/) to route to your redis deployment inside of minikube.

#### Making Code Reusable When Developing

When you are working locally, your redis deployment will be available via `localhost`, but when running in kubernetes, it will be available by the details set in your service definition.

To create reusable code that will work in both environments, we expect you to set two [environment variable in your kubernetes deployment](https://kubernetes.io/docs/tasks/inject-data-application/define-environment-variable-container/) called `REDIS_URL` and `REDIS_PORT` and using [python's `os.getenv` function](https://docs.python.org/3/library/os.html#os.getenv) to utilize the environment variable when available (in kubernetes) otherwise default to `localhost` for host and `6379` for port (when working locally)

#### Verifying your Redis deployment

Redis provides a [command line interface](https://redis.io/docs/connect/cli/) which allows you to interface with a redis instance and view the internals.
While Redis provides [a plethora of commands](https://redis.io/commands/) it will be most helpful for this assignment to focus on [the `KEYS` command](https://redis.io/commands/keys/)

#### Redis Deployment

You should use the official `Redis` image from DockerHub.
You don't need to worry about having incredibly complex deployment setups, and a very minimal configuration to get it running is fine.
In a production environment, we would make this deployment a `StatefulSet`; this requires a lot more configuration, which we won't worry about for this class.

You are expected to have only `1` replica defined for this deployment. Consider why this is important for our simplified case and answer the question in the `SHORT-ANSWER.md`.

### Input Vectorization

***Do not use a `for` loop to iterate through all of your prediction inputs and feed the model one by one.***
Doing this will incur a lot of overhead as your model will have to constantly be brought up and then torn down after doing a single input.
Instead, you can feed the model multiple inputs at the same time and it will return multiple outputs.
This is called input vectorization and will improve performance dramatically when large batches are sent

You are expected to create a method to your input class that will create a vectorized form of the inputs.
You will then be able to run the predict function directly on this vectorized form.
You must apply this to the `multi_predict(...)` function, we have added the base of this function to this submission template.

> [!Caution]
> DO NOT CHANGE THE NAME OF THIS FUNCTION

#### Vectorization Conceptual Example

The key insight is that scikit-learn's `predict()` method accepts a 2D array where each row represents one sample. Instead of calling `predict()` N times for N houses, you should call it once with all N houses.

```python
# Conceptual difference (pseudocode)

# SLOW: N calls to predict()
for each house in houses:
    result = model.predict([[house_features]])  # 1 call per house

# FAST: 1 call to predict()
all_features = [[features_house_1], [features_house_2], ..., [features_house_n]]
results = model.predict(all_features)  # 1 call for all houses
```

> [!TIP]
> Pydantic models maintain field order and can be easily converted to dictionaries. Explore methods like `model_dump()` to help construct your input matrix efficiently.

### Pytest, FastAPI, and Lifespan Events

When working with Lifespan events, we need to change how we run our tests to ensure they are instantiated properly.

When you create a new tests, utilize the following boilerplate:

```python
def test_my_test_that_uses_lifespanned_client():
    with TestClient(app) as lifespanned_client:
        response = lifespanned_client.post(...)
```

### Kubectl and minikube

The following is a non-exhaustive list of commands which you may find useful for utilizing kubectl and minikube

```bash
kubectl version --client
kubectl apply -f <configuration file (.yaml) for deployments | services | namespaces | configMap>
kubectl delete <name of deployment | services | namespace>
kubectl delete --all <deployment | service>
kubectl get all
kubectl get deployments
kubectl get services
kubectl get configmap
kubectl get secrets
kubectl get namespaces
kubectl logs <deployment instance. Example: lab3-deployment-8686ccf5c8-f6j5m>
kubectl <command> --help
kubectl port-forward <deployment instance> <local port>:<remote port>
```

Each of these commands can be followed by `--namespace=w255` if you would like to use it for the w255 namespace, unless you have executed the following command:

```bash
 kubectl config set-context --current --namespace=w255
```

YAML files that you use for deployment should define the specific namespace (i.e. don't leave your YAMLs without the `namespace` argument).

Minikube

```bash
minikube status
minikube start
minikube stop
minikube delete
minikube kubectl version
minikube service list
minikube ip
```

### Networking Requirements

To access your deployments, you will need to utilize one of two systems:

1. Utilize `minikube tunnel` to facilitate networking. You will likely need to run this as a separate subprocess while developing. If you use this method in your run script, you will need to kill this process after you are done with it.
1. Utilize `kubectl port-forward ...` to bridge your local network to your API

Either is acceptable as long as you can make it work consistently.

### Docker Daemons

You should set up your docker daemon to communicate with Minikube so that all build images are accessible from Minikube
`eval $(minikube docker-env)`

You will need to run this command *after* you starting `minikube`

Review the [synopsis described here in the docs](https://minikube.sigs.k8s.io/docs/commands/docker-env/#synopsis) about what this command does.

### Specialized Containers: Init + Readiness + Liveness

Part of your API deployment requires the use of Init Containers, Readiness Probes, and Liveness probes.

#### Init Containers

You should use `busybox` as the base `image`. `busybox` gives you a basic shell to run commands.

You will want one init container that waits for the Redis `service` managed by Kubernetes to be up by asking if the network DNS for that service is available.

After that first init container, you will want a second container which `PING`s the redis deployment through that newly available service to see if the `redis` deployment/application is ready to accept new values in the store. Redis will respond to the `PING` message with a `PONG`. You can ping the container with clever use of netcat (with the command `nc`). Do not use untrusted base images from DockerHub to accomplish this network check.

#### Startup Probe

Like the `HEALTHCHECK` directive in Docker, Kubernetes has a directive to check if the API deployment/pod is ready after startup. Create a `startup` probe that checks if your `/health` endpoint responds to a `GET` request.

The `HEALTHCHECK` directive in docker ***will not work*** in Kubernetes as it is explicitly not supported. You should delete it from your Dockerfile so you don't have any confusion.

#### Readiness Probe

After `startup`, kubernetes checks that a container is ready to be served HTTP traffic by confirming it passes the `readiness` probe.
Create a `readiness` probe that checks if your `/health` endpoint responds to a `GET` request.

#### Liveness Probe

Whereas a `Readiness` probe checks on initial startup, a liveness probe checks to see if the specific pod is still alive/responding.
This should similarly check if the `/health` endpoint to verify if the API is still working (as opposed to stuck in an infinite loop or unresponsive)

### Troubleshooting Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ImagePullBackOff` or `ErrImageNeverPull` | Docker image not available in Minikube | Run `eval $(minikube docker-env)` before `docker build`, then rebuild |
| `CrashLoopBackOff` on API pod | Application failing to start (often Redis connection) | Check logs with `kubectl logs <pod> -n w255`; verify Redis is running first |
| Init container stuck on DNS check | Service name mismatch or service not created | Verify service exists with `kubectl get svc -n w255`; check DNS name format |
| Init container stuck on Redis PING | Redis pod not ready or wrong port | Check Redis pod status; verify service targets correct port (6379) |
| `Connection refused` when testing locally | Tunnel/port-forward not running | Start `minikube tunnel` or use `kubectl port-forward` |
| Tests pass locally, fail in Kubernetes | Different Redis URL in each environment | Verify `REDIS_URL` env var is set in deployment; check `os.getenv()` default |
| API works but cache never hits | Cache decorator not applied or wrong prefix | Verify decorator on endpoint; check Redis keys with `redis-cli KEYS '*'` |
| Probes failing with 404 | Wrong health endpoint path | Probes should use `/lab/health`, not `/health` |

### Debugging Commands

Use these commands to diagnose issues with your Kubernetes deployment:

```bash
# View all resources in your namespace
kubectl get all -n w255

# Check pod status and events (shows why pods are failing)
kubectl describe pod <pod-name> -n w255

# View logs from your API container
kubectl logs <pod-name> -n w255

# View logs from init containers (useful when pod is stuck initializing)
kubectl logs <pod-name> -c init-verify-redis-service-dns -n w255
kubectl logs <pod-name> -c init-verify-redis-ready -n w255

# Execute a shell inside a running container
kubectl exec -it <pod-name> -n w255 -- /bin/sh

# Test Redis connectivity from inside the cluster
kubectl exec -it <redis-pod-name> -n w255 -- redis-cli PING

# View all cached keys in Redis
kubectl exec -it <redis-pod-name> -n w255 -- redis-cli KEYS '*'

# Watch pods in real-time (useful during deployment)
kubectl get pods -n w255 -w
```

> [!TIP]
> Replace `<pod-name>` with the actual pod name from `kubectl get pods -n w255`. Pod names include a random suffix, e.g., `lab3-deployment-8686ccf5c8-f6j5m`.

## Grading

Grades will be given based on the following:

|       **Criteria**       |          **0%**           |                           **50%**                            |                              **90%**                               |                              **100%**                              |
| :----------------------: | :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------: |
|     *Functional API*     |     No Endpoints Work     |                  Some Endpoints Functional                   |                     Most Endpoints Functional                      |                          All Criteria Met                          |
|        *Caching*         |   No Attempt at Caching   |           Caching system instantiated but not used           |       Caching system created but missing some functionality        |                          All Criteria Met                          |
|  *Kubernetes Practices*  | No Attempt at Deployments |         Deployments exist but lack key functionality         |              Kubernetes deployment mostly functional               |                          All Criteria Met                          |
|        *Testing*         |    No Testing is done     | Minimal amount of testing done. No testing of new endpoints. |          Only "happy path" tested and with minimal cases           |                          All Criteria Met                          |
|     *Documentation*      |  No Documentation exists  |                   Very weak documentation                    |                Documentation missing some elements                 |                          All Criteria Met                          |
| *Short-Answer Questions* |  No Questions Attempted   |                 Minimal or incorrect answers                 | Mostly well thought through answers but may be missing some points | Clear and succinct answers that demonstrate understanding of topic |

## Time Expectations

This lab will take approximately ~20 hours. Most of the time will be spent configuring Kubernetes, the deployment, and services, followed by testing to ensure everything is working correctly. Minimal changes to the API are required.
