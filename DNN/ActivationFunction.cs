using System;
using System.Collections.Generic;
using System.Text;

namespace DNN
{
    public interface ActivationFunction
    {
        double activate(double z);
        double derivative(double zprime);
    }
}
