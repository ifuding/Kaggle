#! /bin/bash

in=$1
awk -F',' '
            BEGIN{min_loss=2; min_loss_row = 0;}
            {
                if ($1 < min_loss)
                    {
                        min_loss = $1;
                        min_loss_row = NR;
                    }
            }
            END{print min_loss, min_loss_row}
            ' ${in}
