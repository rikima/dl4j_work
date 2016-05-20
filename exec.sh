#!/bin/sh
cur=$(dirname $0)
pushd $cur

program=com.rikima.dnn.RegressionTest

sbt "run-main $program $*"

popd