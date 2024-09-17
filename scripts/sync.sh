#!/bin/bash

# upload files to a remote server

rootfolder=`dirname $0`/..
hostname=$1
username=$2
destfolder=$3

if [ -z ${hostname} ] || [ -z ${username} ] || [ -z ${destfolder} ];
then
  echo provide the host, username and destination folder as arguments
else
  rsync -avr $rootfolder/src $username@$hostname:$destfolder/xarray_regridder
  rsync -avr $rootfolder/pyproject.toml $username@$hostname:$destfolder/xarray_regridder
  rsync -avr $rootfolder/setup.cfg $username@$hostname:$destfolder/xarray_regridder
fi


