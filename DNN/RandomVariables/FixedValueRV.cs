using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.RandomVariables
{
    public class FixedValueRV : IRandomVariable
    {
        double d;

        public FixedValueRV(double dd)
        {
            d = dd;
        }

        public double next()
        {
            return d;
        }
    }
}
