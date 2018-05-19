<h1 text-align="center">Documentation</h1>
<ul>
<li><p> This is a library created for CNN, to run on opencl. This library is portable and can be used on any device with GPU.</p></li>
  <li><p>The structure of this library is:</p></li>
  <table>
    <tr>
      <td>
        <h3>ConvolutionalNeuralNetwork</h3>
      </td>
      <td>
        The Main object for the convolutional neural network. <br>Contains trainer, compiler, addLayer methods
      </td>
    </tr>
     <tr>
      <td>
        <h3>ConvolutionalLayer</h3>
      </td>
      <td>
        The convolutional Layer object.<br> contains compile, forwardPropagate, backwardPropagate methods
      </td>
    </tr>
    <tr>
      <td>
        <h3>MaxPool</h3>
      </td>
      <td>
        The maxpool layer object.<br> contains compile, forwardPropagate, backwardPropagate methods
      </td>
    </tr>
        <tr>
      <td>
        <h3>Flatten</h3>
      </td>
      <td>
        The Flatten layer object.<br> contains compile, forwardPropagate, backwardPropagate methods
      </td>
    </tr>
     <tr>
      <td>
        <h3>Dense</h3>
      </td>
      <td>
        The Dense layer object.<br> contains compile, forwardPropagate, backwardPropagate methods
      </td>
    </tr>
    
    
  </table>
 </ul>
 <h2>This code is not perfect, and can be made better. If you want to help, contact me email: ihm2015004@iiita.ac.in</h2>
