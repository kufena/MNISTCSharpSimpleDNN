/**

The MIT License (MIT)

Copyright (c) 2019 Andrew Douglas.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other 
liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings 
in the Software.

Line by line description: https://writing.kemitchell.com/2016/09/21/MIT-License-Line-by-Line.html

**/
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
        double min = 0;
        double max = 0;

        public UniformRV(int seed, double min, double max)
        {
            this.min = min;
            this.max = max;

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
