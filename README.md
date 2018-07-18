<!DOCTYPE html>
<html>
<head>
<style>
.container {
    margin-left: 20%;
    margin-right: 20%;
    margin-top: 10px;
    margin-bottom: 10px;
    border: 1px solid gray;
    padding: 25px 50px;
    text-align: left;
}

div > #pic {
  height: 20px;
  align: "center"
}

div > ul > li > img {
  height: 200px;
  align: "center";
}

</style>
</head>
<body>
  <div align="center">

    <h1> Fixed income securities analysis</h1>
    <!--Instalation-->
    <div class="container">
      <h2 align="left">Instalation</h2>
      <p>First you need to install <a href=https://conda.io/docs/user-guide/install/index.html>anaconda</a></p>
      <img class="pic" src="img/instalation.gif" height=200px>
    </div>

    <!--Bootstrapping-->
    <div class="container">
      <h2 align="left">Bootstrapping and Interpolation:</h2>
      <h3>Features</h3>
      <ul>
        <li>Bootstrap strip rates from historical rates given by the US treasury</li>
        <li>Interpolate the term structure</li>
      </ul>
      <h3>Enhancements:</h3>
      <ul>
        <li>Paralel multiprocessing to increase bootsraping speed: 80% reduction in processing time</li>
      </ul>
      <img class="pic" src="img/bootstrap_paralel_processing.png" height=200px>
    </div>

    <!--BDT-->
    <div class="container">
      <h2 align="left">Black-Derman-Toy </h2>
      <h3>Features</h3>
      <ul>
        <li>Fit the model to a term structure and historical volatility data</li>
        <li>Value callable/puttable bonds</li>
      </ul>
      <h3>Enhancements:</h3>
      <ul>
        <li>Load historical rate</li>
        <li>Bootstrao and interpolate the term structure</li>
        <ul>
          <li>Paralel multiprocessing to increase bootsraping speed: 80% reduction in processing time</li>
        </ul>
        <li>Estimate historical volatilities</li>
        <ul>Fit the model</li>
        <li>Use an Adam solver to fit the model</li>
        <ul>
          <li>Calculate numerical gradient with paralel processing, or not</li>
          <li>Visualize the model loss and the ratio of model prices or volatility to real price or volatility</li>
          <img src="img/solver.gif" class="animated_gif" height="200px" >
        </ul>
      </ul>
    </div>

    <!--Vasicek-->
    <div class="container">
      <h2 align="left">Vasicek</h2>
      <h3>Features</h3>
      <ul>
        <li>Fit the model</li>
        <li>Value a callable/puttable zero-coupon bond</li>
      </ul>
    </div>


  </div>
</body>
<html>
