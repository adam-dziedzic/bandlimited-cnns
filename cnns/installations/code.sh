#!/usr/bin/env bash

mkdir code
cd code
git clone git_repository # anonymized
git config credential.helper store  # remember credentials
git pull  # repeat the password