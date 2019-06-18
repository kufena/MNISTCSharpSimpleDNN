using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.RandomVariables
{
    public class UniformRV : IRandomVariable
    {
        Random rand;
        double mult = 0;
        double add = 0;

        public UniformRV(int seed, double min, double max)
        {
            rand = new Random(seed);
            if (min >= max)
                throw new ArgumentException("min after max");

            add = min;
            if (min >= 0)
            {
                mult = max - min;
            }
            else if (max < 0)
            {
                mult = Math.Abs(min) - Math.Abs(max);
            }
            else
            {
                mult = Math.Abs(min) + max;
            }
        }

        public double next()
        {
            return (rand.NextDouble() * mult) + add;
        }
    }
}
