#!/bin/bash
echo -e "Start the full set of benchmarks...\n\n";

echo -e "\nStart package delivery benchmarks...\n\n";
bash run_package_delivery.sh;

echo -e "\nStart spacecraft benchmarks...\n\n";
bash run_spacecraft.sh;

echo -e "\nStart UAV bencharks (4D state space; two spatial dimensions)...\n\n";
bash run_UAV_2D.sh;

echo -e "\nStart UAV bencharks (6D state space; three spatial dimensions)...\n\n";
bash run_UAV_3D.sh;

echo -e "\nBenchmarks completed!";