#! /bin/bash

in=$1

awk -F' ' 'BEGIN{x = 2; y = 0}{if ($NF < x){x = $NF; y = NR}}END{print x, y}' ${in}
