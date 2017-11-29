#! /bin/bash

in=$1
awk -F',' '
            BEGIN{max_row=2; max_auc = 0;}
            NR > 1{
                if ($1 > max_auc)
                    {
                        max_auc = $1;
                        max_row = NR;
                    }
            }
            END{print max_auc, max_row}
            ' ${in}
