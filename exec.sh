#!/bin/sh
cur=$(dirname $0)
pushd $cur

program=jp.ameba.rikima.dnn.RegressionTest

sbt "run-main $program $*"

popd