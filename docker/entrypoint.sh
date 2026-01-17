#!/bin/bash

git config --global --add safe.directory /root/libimmortal

mkdir -p "/root/.config/unity3d/DefaultCompany/Immortal Suffering"
ln -sf /dev/null "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player.log"
ln -sf /dev/null "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player-prev.log"

exec /usr/local/bin/start_xvfb.sh "$@"
