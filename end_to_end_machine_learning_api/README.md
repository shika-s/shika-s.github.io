# Final Project: Full End-to-End Machine Learning API

<!-- markdownlint-disable MD028 -->

<p align="center">
    <!--Hugging Face-->
        <img src="https://user-images.githubusercontent.com/1393562/197941700-78283534-4e68-4429-bf94-dce7ab43a941.svg" width=7% alt="Hugging Face">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--FASTAPI-->
        <img src="https://user-images.githubusercontent.com/1393562/190876570-16dff98d-ccea-4a57-86ef-a161539074d6.svg" width=7% alt="FastAPI">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--REDIS LOGO-->
        <img src="https://user-images.githubusercontent.com/1393562/190876644-501591b7-809b-469f-b039-bb1a287ed36f.svg" width=7% alt="Redis">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--KUBERNETES-->
        <img src="https://user-images.githubusercontent.com/1393562/190876683-9c9d4f44-b9b2-46f0-a631-308e5a079847.svg" width=7% alt="Kubernetes">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--AWS-->
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" width=7% alt="AWS">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--k6-->
        <img src="https://user-images.githubusercontent.com/1393562/197683208-7a531396-6cf2-4703-8037-26e29935fc1a.svg" width=7% alt="K6">
    <!--PLUS SIGN-->
        <img src="https://user-images.githubusercontent.com/1393562/190876627-da2d09cb-5ca0-4480-8eb8-830bdc0ddf64.svg" width=7% alt="Plus">
    <!--GRAFANA-->
        <img src="https://user-images.githubusercontent.com/1393562/197682977-ff2ffb72-cd96-4f92-94d9-2624e29098ee.svg" width=7% alt="Grafana">
</p>

- [Final Project: Full End-to-End Machine Learning API](#final-project-full-end-to-end-machine-learning-api)
  - [Things To Know](#things-to-know)
  - [Base Requirement](#base-requirement)
  - [Project Overview](#project-overview)
  - [Project Objectives](#project-objectives)
  - [Helpful Information](#helpful-information)
    - [Model Background](#model-background)
    - [Pydantic Model Expectations](#pydantic-model-expectations)
    - [Poetry Dependencies](#poetry-dependencies)
    - [Git Large File Storage (LFS)](#git-large-file-storage-lfs)
  - [Submission](#submission)
  - [Grading](#grading)
  - [Time Expectations](#time-expectations)

## Things To Know

> [!Caution]
> You will be deploying 3 services in total; `project`, `lab`, and `redis`. You should expect 3 services, 3 deployments, 2 horizontal pod autoscalers, and a single virtual service that will route based on the path. The only incremental addition is the project. Extend your existing `kustomize` overlay to deploy the `project`. You can copy your `kustomize` scripts from `lab` and extend to add the project. This will stop you from impacting your `lab4` autograder.

> [!Caution]
> You will need to install `git-lfs` in order to pull the model locally. <https://git-lfs.github.com/>. You will be pulling your model from `huggingface`. You can see a tutorial for pulling <https://huggingface.co/docs/hub/en/repositories-getting-started#cloning-repositories>.

> [!Caution]
> `torch` does not support Intel-based Mac's anymore. If you have migrated from an intel-based mac at some point to a new ARM-based Mac you might run into issues. A simple test to verify this is to run the following `poetry run python -c "import platform; print(platform.machine())"` This should show `arm64` for a mac to work as expected. Similarly `arm64` is not supported on Windows, a recent student had issues with a new Microsoft Surface Pro; such computers are not supported by `torch`. If you have this issue the best solution is to launch a virtual machine (either on your computer or in a cloud environment) and do the work on a virtual machine.

> [!NOTE]
> The model for your `lab` was roughly `1 MB`, while the model for the `project` is `~300 MB`. Think about the implications this has for the resources required for your application to run given you have to load your model. Adjust limits accordingly without wasting resources.

> [!NOTE]
> One requirement of the project is that the system is fast when scaling. This implies that new pods are coming online. If the model has to be pulled from `huggingface`, and that model is quite large (`~300 MB`), then it will take a long time for a new pod to come online. During scaling events such as when `k6` is being run this will lead to a significant amount of latency as existing pods won't be able to keep up with the load. You should bake the model into the image similarly to how we have done for the `lab`. The best practice to handle this would be to mount the model from shared storage instead, but that's a lot of extra work for students to understand.

> [!NOTE]
> The training script is provided as a reference. You will not need to run the training yourself as it requires significant GPU resources.

> [!NOTE]
> Your image will be fairly large due to `torch` and the model being roughly 300 MB each. You may run into some networking issues pushing to `ecr`.

## Base Requirement

Expose both your `project` and `lab` over your virtual service to show that we can support multiple services

Since we wrote the `lab` where all endpoints are `/lab` by mounting and the project is setup the same way except for `/project` we can simply route based on the url path to a particular service.

Provided is a minimal Virtual Service definition that you can extend for multiple matches.

```{yaml}
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: external-access
  namespace: winegarj
spec:
  gateways:
    - istio-ingress/winegarj-gateway
  hosts:
    - winegarj.mids255.com
  http:
    - match:
        - uri:
            prefix: "/lab"
      route:
        - destination:
            host: lab-prediction-service
            port:
              number: 8000
```

## Project Overview

The goal of `project` is to take everything you have learned in this class and deploy a fully functional prediction API accessible to end users.

You will:

- Utilize `Poetry` to define your application dependencies
- Package up an existing NLP model ([DistilBERT](https://arxiv.org/abs/1910.01108)) for running efficient CPU-based sentiment analysis from `HuggingFace`
- Adjust your `redis` cache prefix key to `fastapi-cache-project`
- Create a `FastAPI` application to serve prediction results from user requests
- Test your application with `pytest`
- Utilize `Docker` to package your application as a logic unit of compute
- Cache results with `Redis` to protect your endpoint from abuse
- Deploy your application to `AWS` with `Kubernetes`
- Use `k6` to load test your application
  - `k6 run -e NAMESPACE=${NAMESPACE} --summary-trend-stats "min,avg,med,max,p(90),p(95),p(99),p(99.99)"  load.js`
- Use `Grafana` to visualize and understand the dynamics of your system

Below is an outline of the overall architecture:

```mermaid
flowchart TB
subgraph Entire [ ]
    subgraph Userland [ ]
        User(User)
        Developer(Developer)
    end
    subgraph AWS [AWS]
        subgraph Account [Account]
            subgraph ECR [Elastic Container Registery]
                subgraph repo [Container Repo]
                    i1(image tag:4f925d7)
                    i2(image tag:29a727c):::fa
                end
            end
            subgraph k8s [Elastic Kubernetes Service]
                subgraph istio [Namespace: istio-ingress]
                    gw(your-name-gateway)
                end
                subgraph subgraph_padding1 [ ]
                    subgraph cn [Namespace: your-name-here]
                        direction TB
                        subgraph subgraph_padding2 [ ]
                            NPS2(ClusterIP: lab-prediction-service):::nodes
                            NPS3(ClusterIP: project-prediction-service):::nodes
                            subgraph PD [Lab Deployment]
                                direction TB
                                IC1(Init Container: verify-redis-dns)
                                IC2(Init Container: verify-redis-ready)
                                FA(Lab FastAPI Container):::fa

                                IC1 --> IC2 --> FA
                            end
                            subgraph ProD [Project Deployment]
                                direction TB
                                IC3(Init Container: verify-redis-dns)
                                IC4(Init Container: verify-redis-ready)
                                FA2(Project FastAPI Container):::fa

                                IC3 --> IC4 --> FA2
                            end
                            NPS1(ClusterIP: redis-service):::nodes
                            RD(Redis Deployment)

                            VS(VirtualService)

                            VS <--> NPS2
                            VS <--> NPS3

                            NPS1 <--->|Port 6379| PD
                            NPS1 <-->|Port 6379| ProD
                            NPS1 <-->|Port 6379| RD
                            NPS2 <-->|Port 8000| PD
                            NPS3 <-->|Port 8000| ProD
                        end
                    end
                end
                i2 -..- FA
                i1 -...- FA2
            end
        end
    end
end
gw <---> User
VS <--> gw

Developer -.->|aws sts get-caller-identity --profile ucberkeley-sso | AWS
Developer -.->|aws sts get-caller-identity --profile ucberkeley-student | Account
Developer -.->|"aws ecr get-login-password --region us-west-2 --profile ucberkeley-student | docker login --username AWS --password-stdin 650251712107.dkr.ecr.us-west-2.amazonaws.com"| ECR
Developer -.->|aws eks update-kubeconfig --name datasci255-eks --profile ucberkeley-student | k8s
Developer -->|docker push| repo

classDef nodes fill:#68A063
classDef subgraph_padding fill:none,stroke:none
classDef inits fill:#cc9ef0
classDef fa fill:#00b485

style cn fill:#B6D0E2;
style RD fill:#e6584e;
style PD fill:#FFD43B;
style ProD fill:#FFD43B;
style k8s fill:#b77af4;
style AWS fill:#00aaff;
style Account fill:#ffbf14;
style ECR fill:#cccccc;
style repo fill:#e7e7e7;
style Userland fill:#ffffff,stroke:none;
style Entire fill:#ffffff,stroke:none;

class subgraph_padding1,subgraph_padding2 subgraph_padding
class IC1,IC2,IC3,IC4 inits
```

## Project Objectives

- [ ] Write pydantic models to match the specified ***input*** model

    ```javascript
    {
        "text": ["example 1", "example 2"]
    }
    ```

- [ ] Write pydantic models to match the specified output model

    ```javascript
    {
        "predictions":
        [
            [
                {
                    "label":"POSITIVE",
                    "score":0.7127904295921326
                },
                {
                    "label":"NEGATIVE",
                    "score":0.2872096002101898
                }
            ],
            [
                {
                    "label":"POSITIVE",
                    "score":0.7186233401298523
                },
                {
                    "label":"NEGATIVE",
                    "score":0.2813767194747925
                }
            ]
        ]
    }
    ```

- [ ] Pull the [following model](https://huggingface.co/winegarj/distilbert-base-uncased-finetuned-sst2) locally to allow for loading into your application. Put this at the root of your project directory for an easier time.
  - [x] Add the model files to your `.gitignore` since the file is large, and we don't want to manage `git-lfs` and incur costs for wasted space. `HuggingFace` is hosting the model for us.
- [ ] Create and execute `pytest` tests to ensure your application is working as intended
- [ ] Build and deploy your application locally (Hint: Use `kustomize`)
- [ ] Push your image to `ECR`.
  - [ ] Use a prefix based on your namespace, and call the image `project`
- [ ] Deploy your application to `EKS` similar to lab 4
  - [ ] Your project deployment must be named `project-api-deployment`
  - [ ] Your project service must be named `project-prediction-service`
  - [ ] Include the same init containers (`init-verify-redis-service-dns`, `init-verify-redis-ready`) as your lab deployment
  - [ ] Include readiness, liveness, and startup probes pointing to `/project/health`
  - [ ] **Make sure to adjust your virtual service to expose both `project` and `lab`**
- [ ] Run `k6` against your application with the provided `load.js`
- [ ] Capture screenshots of your `grafana` dashboard for your service/workload during the execution of your `k6` script
- [ ] Feel extremely proud about all the learning you went through over the semester and how this will help you develop professionally and enable you to deploy an API effectively during your capstone. There is much to learn, but getting the fundamentals are key.

## Helpful Information

### Model Background

Please review the `train.py` to see how the model was trained and pushed to `HuggingFace` as an artifact store for models and their associated configuration.
This model took 5 minutes to transfer learn on 2x A4000 GPUs with a 256 batch size, taking 15 GB of memory on each GPU.

Training on CPUs would likely have taken several days. The given implementation allows for maximum text sequences of `512` tokens for each input.
***Do not try to run the training script on your local machine.***

Model loading examples are provided in `example.py`. In this file, we load the model from a local directory; however, loading from `HuggingFace` directly is extremely inefficient given the size of the underlying model (~300 MB) for a production environment.
We will pull down the model locally as part of our build process.

Model prediction pipelines are included in the `transformers` API provided by `HuggingFace,` which dramatically reduces the complexity of the Inferencing application.
An example is provided in `mlapi/example.py` and is instrumented already in your `main.py` application.

### Pydantic Model Expectations

We provide you with a pytest file, `test_mlapi.py`, which has the structure of how you should design your pydantic models.
You will have to do some reverse engineering so that your model matches our expectations.

> [!IMPORTANT]
> The template code contains intentional mismatches with the test expectations. Read the test file carefully and compare it against the provided template—you will need to fix these discrepancies.

### Poetry Dependencies

Do not run `poetry update` it will take a long time due to the handling of `torch` dependencies.
Do a `poetry install` instead.

### Git Large File Storage (LFS)

You might need to install `git lfs` <https://git-lfs.github.com/>

### Baking the Model into Your Docker Image

To ensure fast scaling, you must copy the model into your Docker image at build time. This prevents pods from having to download the model from HuggingFace during startup.

> [!NOTE]
> Baking the model into the image is not best practice. In production environments, you would typically mount the model from shared storage (e.g., EFS, S3). However, setting up shared storage adds significant complexity, so for this project we choose to bake the model into the image.

### Resource Limits

The project model is significantly larger than the lab model (~300 MB vs ~1 MB). You will need to adjust your resource requests and limits accordingly.

> [!WARNING]
> The cluster uses Gatekeeper policies to enforce resource constraints. If your resource requests are too high, your pods will be rejected. You need to find the right balance: enough resources for the model to load and run, but not so much that Gatekeeper blocks your deployment.

### Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Pod stuck in `Pending` state | Resource requests exceed Gatekeeper limits | Reduce `requests.memory` and `requests.cpu` in your deployment |
| Pod OOMKilled | Model exceeds memory limit | Increase `limits.memory` (but stay within Gatekeeper constraints) |
| Slow pod startup during k6 | Model downloading from HuggingFace | Bake the model into your Docker image |
| Tests fail with key mismatch | Template code has intentional bugs | Compare test assertions against your response structure |
| `ImagePullBackOff` | Image not pushed to ECR or wrong tag | Verify image exists: `aws ecr describe-images --repository-name <namespace>/project` |
| High latency during load test | Not enough replicas or cache not working | Check HPA status and verify Redis cache hit rate |
| 504 Gateway Timeout | Prediction taking too long | Check resource limits, consider increasing replicas |

### Deployment Verification Checklist

Given the complexity of this deployment (3 services, 3 deployments, 2 HPAs, 1 VirtualService), use this checklist to verify your setup is complete:

| Resource | Expected | Verification Command |
|----------|----------|----------------------|
| Deployments | 3 (`lab-api-deployment`, `project-api-deployment`, `redis-deployment`) | `kubectl get deployments -n $NAMESPACE` |
| Services | 3 (`lab-prediction-service`, `project-prediction-service`, `redis-service`) | `kubectl get services -n $NAMESPACE` |
| HPAs | 2 (lab, project) | `kubectl get hpa -n $NAMESPACE` |
| VirtualService | 1 (routes both `/lab` and `/project`) | `kubectl get virtualservice -n $NAMESPACE` |
| Init Containers | 2 per app deployment (`init-verify-redis-service-dns`, `init-verify-redis-ready`) | `kubectl get deployment project-api-deployment -o jsonpath='{.spec.template.spec.initContainers[*].name}'` |
| Probes | readiness, liveness, startup on `/project/health` | `kubectl describe deployment project-api-deployment -n $NAMESPACE` |
| Running Pods | At least 3 | `kubectl get pods -n $NAMESPACE` |

## Submission

All code will be graded off your repo's `main` branch and `EKS` deployment.
No additional forms or submission processes are needed.

## Grading

All items are conditional on a `95%` cache rate, and after a `10 minute` sustained load:

|                  **Criteria**                  |                          **0%**                          |                            **50%**                            |                        **90%**                         |                   **100%**                   |
|:---------------------------------------------: |:-------------------------------------------------------: |:------------------------------------------------------------: |:-----------------------------------------------------: |:-------------------------------------------: |
| *Functional API*                               | No Endpoints Work                                        | Some Endpoints Functional                                     | Most Endpoints Functional                              | All Criteria Met                             |
| *Caching*                                      | No Attempt at Caching                                    | Caching system instantiated but not used                      | Caching system created but missing some functionality  | All Criteria Met                             |
| *Kubernetes Practices*                         | No Attempt at Deployments                                | Deployments exist but lack key functionality                  | Kubernetes deployment mostly functional                | All Criteria Met                             |
| *Testing*                                      | No Testing is done                                       | Minimal amount of testing done. No testing of new endpoints.  | Only "happy path" tested and with minimal cases        | All Criteria Met                             |
| *Passing Provided Tests*                       | Pydantic model does not adhere to our given pytest file  | Pydantic model somewhat passes pytest file                    | Pydantic model mostly passes pytest file               | All Criteria Met                             |
| *Model Loading*                                | Model loads from hugging face on API instantiation       | N/A                                                           | N/A                                                    | Model is loaded into the container at build  |
| *Predict Endpoint Performance*                 | Endpoint performs at 1 request/second                    | Endpoint performs at 5 requests/second                        | Endpoint performs at 9 request/second                  | Endpoint performs at 10 requests/second      |
| *Predict Endpoint Latency @ 10 Virtual Users*  | p(99) < 10 seconds                                       | p(99) < 5 seconds                                             | p(99) < 3 seconds                                      | p(99) < 2 seconds                            |

## Time Expectations

This project will take approximately ~10 hours.
