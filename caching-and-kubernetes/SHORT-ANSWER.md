# Short Answer Questions

## What are the benefits of caching?

Caching offers several benefits including performance, scalability, reduced load, better user experience, network efficiency, and cost savings. Cacing takes the load off the backend systems like databases and servers. 

## What is the difference between Docker and Kubernetes?

Docker is a containerization platform. Kubernetes is a container orchestration platform. Docker helps to package an application and all its dependencies into a lightweight, portable container that runs consistently across environments. Kubernetes manages and coordinates many containers across multiple machines. Docker creates and runs individual contianers. Kubernetes decides where containers run, scales them up or down, restarts them if they fail, and handles networking between them.  

## What does a kubernetes `deployment` do, how is this different from a `service`?

Kubernetes deployment manages the lifecycle of the application pods. Its respopnsible for declaring what should be running: which container image, how many replicas, resource limits, environment variables, and update strategy. Kubernetes Service manages network access to the pods. Pods are ephemeral, they get created, destroyed and rescheduled with new  IP addresses constantly. A service provides a stable endpoint that routes traffic to the right set of pods. This separation of concerns is a core Kubernetes design principle, it lets us scale, update, or replace our application without changing how other services discover and communicate with it. 

## In our simplified use case, why should we only have 1 redis replica instead of 3?
In our simplified use case, we are using the redis cache to store the predicted results from the /bulk-predict endpoint. So, in case the redis cache is not available, the worst-case scenario would be that the model would have to predict again, causing a small delay to the user. For this reason, the  complexity involved with managing 3 Redis replica is not warranted. 

