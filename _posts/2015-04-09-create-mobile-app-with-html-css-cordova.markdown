---
layout: post
title:  "Create a mobile app from HTML/CSS with Cordova framework"
date:   2015-04-09 23:00:51
categories: mobile
---

#Create the app 
		
Create the app project folder and add the IOS and Android platforms

	cordova create dir_name com.domain.my-app app_name
	cd dir_name
	cordova platform add ios
	cordova platform add android

#Add the native plugins you wish


####Camera 

To launch the native camera :

	cordova plugin add https://git-wip-us.apache.org/repos/asf/cordova-plugin-file-transfer.git
	cordova plugin add https://git-wip-us.apache.org/repos/asf/cordova-plugin-camera.git

If you want to use JPEG with GPS localization, comment the following lines :

	NSDictionary *controllerMetadata = [info objectForKey:@"UIImagePickerControllerMediaMetadata"];
	if (controllerMetadata) { self.data = data; self.metadata = [[NSMutableDictionary alloc] init];
	NSMutableDictionary *EXIFDictionary = [[controllerMetadata objectForKey:(NSString *)kCGImagePropertyExifDictionary]mutableCopy]; 
	if (EXIFDictionary) [self.metadata setObject:EXIFDictionary forKey:(NSString *)kCGImagePropertyExifDictionary]; 
	[[self locationManager] startUpdatingLocation]; return; }

as described [in this article](http://stackoverflow.com/questions/17253139/how-to-remove-location-services-request-from-phonegap-ios-6-app).

####Splashscreen

To manage the splashscreen

	cordova plugin add org.apache.cordova.splashscreen


####Google Analytics 

I recommand this plugin : 

	cordova plugin add https://github.com/danwilson/google-analytics-plugin.git

danwilson/google-analytics-plugin has not been updated for a while.

####Network informations

	cordova plugin add org.apache.cordova.network-information

If trouble during iOS compilation, add `SystemConfiguration.framework` in *Build Phases > Link Binary With Libraries*.

####Facebook

Due to a problem, you need to download a local copy

	cd ..
	git clone https://github.com/Wizcorp/phonegap-facebook-plugin.git
	cd dir_name
	cordova plugins add ../../phonegap-facebook-plugin/ --variable APP_ID="xxx" --variable APP_NAME="yyy"

In *config.xml*, add :

	<gap:plugin name="com.phonegap.plugins.facebookconnect" version="0.9.0">
		<param name="APP_ID" value="xxx" />
		<param name="APP_NAME" value="yyy" />
	</gap:plugin>

The APP_ID is the ID provided by Facebook App Center. Create a development key : 

	keytool -exportcert -alias androiddebugkey -keystore ~/.android/debug.keystore | openssl sha1 -binary | openssl base64	

with password `android` and production key : 

	keytool -exportcert -alias alias_name -keystore ../../certificates-ADMIN/android/my-release-key.keystore | openssl sha1 -binary | openssl base64

Add both keys to the Facebook App Center.

####Deep linking / custom URL schemes

	cordova plugin add https://github.com/EddyVerbruggen/LaunchMyApp-PhoneGap-Plugin.git --variable URL_SCHEME=your_sheme

Hence, `your_sheme://path` will launch your app on the mobile device if it has been installed.

If you want to have it work with `http://my_domain.com/view` for example, add an INTENT action for the path `/view` by adding in the `<activity>` of *config.xml* :

	<intent-filter>
		<action android:name="android.intent.action.VIEW"/>
		<category android:name="android.intent.category.DEFAULT"/>
		<category android:name="android.intent.category.BROWSABLE"/>
		<data android:scheme="http" android:host="my_domain.com" android:path="/view" />
	</intent-filter>

####Notification Push

You can use the standard plugin

	cordova plugin add https://github.com/phonegap-build/PushPlugin.git

or I would recommend you the [Radium One Plugin](https://github.com/radiumone/r1-connect-demo-phonegap) that gives you a great interface to push and segment your users.

	cordova plugin add https://github.com/radiumone/r1-connect-demo-phonegap

####Other plugins : Clipboard, device, file, file-transfer, statusbar, inappbrowser

	cordova plugins add com.verso.cordova.clipboard
	cordova plugins add org.apache.cordova.device
	cordova plugins add org.apache.cordova.file
	cordova plugins add org.apache.cordova.file-transfer
	cordova plugins add org.apache.cordova.inappbrowser
	cordova plugins add org.apache.cordova.statusbar


#Run 

	cordova run android
	cordova run ios


#Compile the app for release

####Set the version

In *config.xml*

####IOS

Be careful to change “apns_sandbox” by “apns” in your Javascript code if you use Notification Push plugin, in order to use the Apple notification server for production apps.

	cordova build ios	

Launch *Selectionnist.xcodeproj* with XCODE.

In *Build Settings > Code signing*, choose the right mobile provisionning (need to configure accounts in XCODE before).

In *General > App Icons & Launch Images*, verify images.

Then select *Product > Archive* in the menu bar, and submit your app to Apple.

####Android

In *platforms/android/AndroidManifest.xml* add

	<application `*`android:debuggable="false"`*` android:hardwareAccelerated="true" android:icon="@drawable/icon" android:label="@string/app_name">

Compile

	cordova build --release android

Sign the APK

	jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore ../../certificates-ADMIN/android/my-release-key.keystore  platforms/android/ant-build/CordovaApp-release-unsigned.apk alias_name

Align 

	zipalign -v 4 platforms/android/ant-build/CordovaApp-release-unsigned.apk platforms/android/ant-build/CordovaApp-release.apk

Submit the *CordovaApp-release.apk* to Google Play Publish.

**And here you're!**

