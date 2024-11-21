
"use strict";

(function($){

	$.extend(verge);

	//*********** Sticky Menu Begin *********//	

	var stickyHeader={
		init:function(){
			this.$header=$('#header');
			this.$contentWrapper=$('#wrapper');
			this.headerHeight=this.$header.outerHeight();
			this.absFlag=(this.$header.css('position')!='absolute')?true:false;
			this.threshold=this.$header.data('sticky-threshold')?this.$header.data('sticky-threshold'):0;
			this.stickyFlg=false;
			this.$pageHead=$('.page-head');
			this.setHeadBottom();
			this.bindUIActions();

		},
		bindUIActions:function(){
			var self=this;
			$(document).on('scroll',function(){
				var scrollPos=$(this).scrollTop();
				if (scrollPos>self.headBottom+self.threshold){
					self.changeState('stick');
				}
				else if (scrollPos<self.headBottom){
					self.changeState('unstick');
				}
			});

			$(window).on('debouncedresize',function(){
				self.setHeadBottom();
				self.headerHeight=self.$header.outerHeight();
			});
		},
		changeState:function(state){
			var self=this;

			if ($.og.$window.width()<=768 || $.og.$header.hasClass('mobile-menu')) return;

			if(state=='stick' && !self.stickyFlg){
				self.$header.addClass('is-sticky');
				if (self.absFlag)
					self.$contentWrapper.css({marginTop:'+='+self.headerHeight});
				self.stickyFlg=true;
				$(document).trigger('sticky-change');
			}else if(state=='unstick' && self.stickyFlg){
				self.$header.removeClass('is-sticky');
				if (self.absFlag)
					self.$contentWrapper.css({marginTop:''});
				self.stickyFlg=false;
				$(document).trigger('sticky-change');

			}
		},
		setHeadBottom:function(){
			var self=this;

			if (self.$pageHead.length){
				if (self.$pageHead.outerHeight()>70){
					self.headBottom=self.$pageHead.outerHeight()+self.$pageHead.offset().top;
				}else{
					self.headBottom=$(window).height();
				}
			}else{
				self.headBottom=self.$header.height()+self.$header.offset().top;
			}
		}
	};
	//*********** Sticky Menu End *********//	

	
	//*********** Primary Menu Begin*********//	

	var olmenu = {
		init:function($elem){
			this.$elem = $elem;
			this.$links = this.$elem.find("a");
			this.$hParent=this.$elem.parents('.head-main, .nav-row');//The parent that determines height
			this.prepare();
			this.setMobile(this.isMobileActive());
			this.bindUIActions();
			this.toggleAnimation={duration:300,easing:"easeInOutQuint"};
			
		},
		prepare:function(){
			var self = this;
			self.$links.attr('title','');
		},
		bindUIActions:function(){
			var self = this;

			$(window).on('debouncedresize',function(){
				self.destroy();
				self.setMobile(self.isMobileActive());
			});

			$.og.$body.on('click','.ol-mobile-trigger',function(){
				$(this).toggleClass("is-active");
				self.$elem.stop().slideToggle(self.toggleAnimation);
			})
			.on('click','.mobile-menu .menu-item-has-children > a',function(e){
				e.preventDefault();
				var $parent=$(this).parent();
				$parent.toggleClass('is-open').children('.sub-menu').stop().slideToggle(self.toggleAnimation);
			});


		},
		setMobile:function(mobileFlag){
			var self=this;
			if (mobileFlag){
				$.og.$header.addClass('mobile-menu');
			}else{
				$.og.$header.removeClass('mobile-menu');
			}
		},
		isMobileActive:function(){
			var self=this;

			// mobile device?
			if ( $.browser.mobile ) return true;

			if ( $.og.$window.width()<=1200 ) return true;
				
		},
		destroy:function(){
			$.og.$header.removeClass('mobile-menu');
		}

	};
	//*********** Primary Menu End *********//	

	
	//*********** Logo Handler Begin *********//

	var logoHandler={
		init:function(){
			this.$wrapper = $('.logo-wrapper');
			this.$imgs=this.$wrapper.find('img');

			if (this.$imgs.length<=1) return false;

			this.$logoLight=this.$imgs.filter('.logo-light');
			this.$logoDark=this.$imgs.filter('.logo-dark');

			this.bindUIActions();
			this.changeSrc(this.decision());
		},
		bindUIActions:function(){
			var self=this;
			$(document).on('sticky-change',function(){
				self.changeSrc(self.decision());
			})
		},
		decision:function(){
			var self=this;
			var mode=$.og.$header.hasClass('dark')?'dark':'light';

			if ($.og.$header.hasClass('is-sticky')){
				if ($.og.$header.hasClass('sticky-dark')){
					mode='dark';
				}else if ($.og.$header.hasClass('sticky-light')){
					mode='light';
				}
			}
			return mode;
			
		},
		changeSrc:function(mode){
			var self=this;
			if (mode=='light'){
				self.$logoDark.hide();
				self.$logoLight.css('display','inline-block');
			}else{
				self.$logoDark.css('display','inline-block');
				self.$logoLight.hide();
			}
		}

	};
	//*********** Logo Handler End *********//


	//*********** Set-bg Images Begin *********//
	var setBg = {
		init:function($elem){
			var imgUrl = this.getImageInside($elem);
			if ( imgUrl ){
				$elem.css('background-image', 'url(' + imgUrl + ')');
			}
			
		},
		destroy:function($elem){
			$elem.css('background-image','');
			$elem.find('img.set-me').show();
		},
		getImageInside:function($elem){
			//var $insideImage = $e.find('img.set-me').hide();
			var $dataImage = $elem.data('img-src');

			if ( $dataImage )
				return $dataImage;
			else {
				var $insideImage = $elem.find('img.set-me').first();
				$insideImage=($insideImage.length)?$insideImage:$elem.find('img').first();

				if (!$insideImage.length){
					return false
				}

				$insideImage.hide();
				return $insideImage.attr('src')
			}
		}
	};
	//*********** Set-bg Images End *********//


	//*********** Handle Particles Begin *******//
	var particles = {
		init:function($elem){
			this.$elem = $elem;
			this.prepare();
			this.run();
		},
		run:function(){
			var self = this;
			var isDark = self.$elem.hasClass('dark') || self.$elem.hasClass('dark-wrapper') ? true : false;

			var particleColor = "#ffffff";
			var linkcolor = "#ffffff";

			if ( ! isDark ){
				particleColor = "#000000";
				linkcolor = "#000000";
			}

			particlesJS(
				self.$elem.ID, 
				{
				  "particles": {
				    "number": {
				      "value": 80,
				      "density": {
				        "enable": true,
				        "value_area": 800
				      }
				    },
				    "color": {
				      "value": particleColor
				    },
				    "shape": {
				      "type": "circle",
				      "stroke": {
				        "width": 0,
				        "color": "#000000"
				      },
				      "polygon": {
				        "nb_sides": 5
				      },
				      "image": {
				        "src": "img/github.svg",
				        "width": 100,
				        "height": 100
				      }
				    },
				    "opacity": {
				      "value": 0.1,
				      "random": false,
				      "anim": {
				        "enable": false,
				        "speed": 1,
				        "opacity_min": 0.2,
				        "sync": false
				      }
				    },
				    "size": {
				      "value": 3,
				      "random": true,
				      "anim": {
				        "enable": false,
				        "speed": 40,
				        "size_min": 0.1,
				        "sync": false
				      }
				    },
				    "line_linked": {
				      "enable": true,
				      "distance": 150,
				      "color": linkcolor,
				      "opacity": 0.2,
				      "width": 1
				    },
				    "move": {
				      "enable": true,
				      "speed": 0.8,
				      "direction": "none",
				      "random": false,
				      "straight": false,
				      "out_mode": "out",
				      "bounce": false,
				      "attract": {
				        "enable": false,
				        "rotateX": 600,
				        "rotateY": 1200
				      }
				    }
				  },
				  "interactivity": {
				    "detect_on": "canvas",
				    "events": {
				      "onhover": {
				        "enable": true,
				        "mode": "grab"
				      },
				      "onclick": {
				        "enable": true,
				        "mode": "push"
				      },
				      "resize": true
				    },
				    "modes": {
				      "grab": {
				        "distance": 158,
				        "line_linked": {
				          "opacity": 0.5
				        }
				      },
				      "bubble": {
				        "distance": 400,
				        "size": 40,
				        "duration": 2,
				        "opacity": 8,
				        "speed": 3
				      },
				      "repulse": {
				        "distance": 200,
				        "duration": 0.4
				      },
				      "push": {
				        "particles_nb": 4
				      },
				      "remove": {
				        "particles_nb": 2
				      }
				    }
				  },
				  "retina_detect": true
				}
			);


		},
		prepare:function(){
			var self = this;
			var ID = self.$elem.attr("id");
			self.$elem.ID = ID;
			if ( ! ID ){
				var randId = "ol-particles-"+makeid();
				self.$elem.attr("id", randId);
				self.$elem.ID = randId;
			}
		}

	};

	//*********** Handle Particles End *******//
	

	//*********** Accordion Handler Begin *******//
	/* Simple Accordion 
	/* Toggle free style accordion
	/* Side navigation accordion & toggle style*/

	var accordion={
		init:function($elem,options){
			this.$elem=$elem;

			var defaultOptions={
				itemSelector:'.ac-item',
				headSelector:'.item-head',
				bodySelector:'.item-body',
				activeClass:'open',
				initActiveClass:'open',
				addToggleElem : true,
				toggleElemClass : '.item-head',
				toggleEl:'<i class="ol-toggle-icon">'
			};

			this.options=$.extend( {}, defaultOptions, options);

			this.$items=this.$elem.children(this.options.itemSelector);
			this.SingleToggleFlag=this.$elem.hasClass('toggle-free')?false:true;
			
			this.prepare();
			this.bindUIActions();

		},
		prepare:function(){
			var self=this;
			var op=this.options;
			var $openItems=self.$items.filter('.'+op.initActiveClass);

			if ($openItems.length){

				//Only one toggle can be open by default
				if (self.SingleToggleFlag && $openItems.length>1){
					$openItems.removeClass(op.initActiveClass);
					$openItems=$openItems.first().addClass(op.initActiveClass);
				}
				$openItems.addClass(op.activeClass);
				$openItems.children(op.bodySelector).show();
			}//else{
				//do we want to force open the first tab??
			//}

			if ( op.addToggleElem ){
				self.$elem.find(op.toggleElemClass).append($(op.toggleEl));
			}
			
		},
		bindUIActions:function(){
			var self=this,
				op=this.options;

			self.$elem.on('click',op.headSelector,function(e){

				var $this=$(this),
					$parent=$this.parent(),
					$itemBody=$this.next(op.bodySelector);

				if ($itemBody.length){
					e.preventDefault();
					//There is a content section that should be shown
					if (self.SingleToggleFlag && !$parent.hasClass(op.activeClass)){
						var $openItems=$parent.siblings('.'+op.activeClass).removeClass(op.activeClass);
						self.toggleElem($openItems.children(op.bodySelector));
					}

					self.toggleElem($itemBody);
					$parent.toggleClass(op.activeClass);
				}

			});
		},
		toggleElem:function($elem){
			$elem.slideToggle({ duration: 400, easing: "easeInOutQuart" });
		}

	};

	//*********** Accordion Handler End *******//
	

	//*********** Tabs Begin *******//
	var tabs={
		init : function($elem){

			this.$elem=$elem;
			this.$bodyItems=$elem.find('.tab-pane');
			this.$headItems=$elem.find('.tab-navigation').children('li');

			this.prepare();
			this.bindUIActions();
			
			
		},
		prepare:function(){
			var self=this;

			var $activeHead=self.$headItems.filter('.active').first();

			if ($activeHead.length<1){
				self.$bodyItems.removeClass('active').eq(0).addClass('active');
				self.$headItems.removeClass('active').eq(0).addClass('active');
			}

		},
		bindUIActions : function(){
			var self = this;

			self.$elem.on('click','.tab-navigation a', function(e){

				e.preventDefault();
				
				var $this=$(this),
					$parent = $(this).parent(),
					index = $parent.index();

				if ($parent.hasClass('active')) return false;

				self.$headItems.removeClass('active');
				$parent.addClass('active');
				self.$bodyItems.removeClass('active').eq(index).addClass('active');
			});
		}
	};
	//*********** Tabs End *******//
	

	//*********** Timelines Begin *******//
	var olTimeline={
		init: function($elem){
			this.$olTimeline = $elem;
			this.$bodyItems=this.$olTimeline.find('.tl-item');
			this.$sections=$();
			this.activeIndex=0;
			
			var self=this;

			


			this.bindUIActions();
			this.prepare();
			
		},
		bindUIActions:function(){
			var self=this;	
		},
		prepare:function(){
			var self=this;

			self.arrangeSections();
			self.$sections.each(function(){

				var $this=$(this),
					$itemSec=$this.find('.item-section').first();

				if ($itemSec.parent().hasClass('with-thumb')){
					$this.addClass('with-thumb');
				}
				if ($itemSec.parent().hasClass('with-icon')){
					$this.addClass('with-icon');
				}

				$this.append($itemSec);	
				$itemSec.stick_in_parent({
					offset_top:window.olStickyOffset+30
					
				}).on("sticky_kit:bottom", function(e) {
					$itemSec.addClass('is_bottom');
			  	}).on('sticky_kit:unbottom',function(){
			  	  	$itemSec.removeClass('is_bottom');	
			  	});

			})
		},
		arrangeSections:function(){
			var self=this,
				lastSection,counter=0,
				$tempSec=$();

			self.$bodyItems.each(function(){
				var $this=$(this),
					itemSec=$this.find('.item-section').text();
				if ((lastSection==undefined || itemSec!=lastSection) && itemSec){
					if ($tempSec.length>0){
						var $wrapper=$('<div></div>').addClass('tl-section');
						$tempSec.wrapAll($wrapper);
						self.$sections=self.$sections.add($tempSec.parent());
						
						$tempSec=$();
					}
					$tempSec=$tempSec.add($this);
					lastSection=itemSec;
					counter++;
					//$this.attr('data-section-index',counter++);
					//self.$sections=self.$sections.add($this);
					return true;
				}else if (lastSection!=undefined){
					$tempSec=$tempSec.add($this);
				}


			});

			if ($tempSec.length>0){
				var $wrapper=$('<div></div>').addClass('tl-section');
				$tempSec.wrapAll($wrapper);
				self.$sections=self.$sections.add($tempSec.parent());
			}

			self.sectionsNum=counter-1;
		}
	};

	//*********** Timelines End *******//

	//*********** Timeline Tabs Begin *******//
	var olTimeTab={
		init:function($elem){
			this.$elem=$elem;
			this.$head=this.$elem.find('.tl-head');
			this.$body=this.$elem.find('.tl-content');
			this.$headItems=this.$head.children();
			this.$bodyItems=this.$body.children('.tl-item');
			this.itemsNumber=this.$headItems.length;
			this.startFrom=this.$elem.data('start-from')?this.$elem.data('start-from'):'center';
			this.fadeEdge=this.$elem.data('fade-edge')?this.$elem.data('fade-edge'):true;

			this.prepare();

			if (this.fadeEdge)
				this.$elem.addClass('with-fader');
			this.bindUIActions();
			
		},
		prepare:function(){
			var self=this;

			switch (self.startFrom){
				case 'center':
					self.activeIndex=Math.floor(self.itemsNumber/2);
					break;
				case 'first':
					self.activeIndex=0;
					break;
				case 'last':
					self.activeIndex=self.itemsNumber-1;
					break;
				default:
					self.activeIndex=parseInt(self.startFrom);
					break;
			}


			

			self.$headClone=self.$head.clone().addClass('tl-head-clone').prependTo(self.$elem);
			self.$headCloneItems=self.$headClone.children();
			self.set();
			self.goTo(self.activeIndex);

		},
		set:function(){
			var self=this;

			self.elemHeight=self.$elem.height();
			self.elemTopPos=self.$elem.offset().top;
			self.headCenterPos=self.elemTopPos+(self.elemHeight/2)-10;
			self.headItemHeight=self.$headItems.first().outerHeight();
			self.headItemBoundry=self.headItemHeight+parseInt(self.$headItems.first().css('margin-bottom'));
			self.$bodyItems.outerHeight(self.elemHeight);
		},
		goTo:function(idx){
			var self=this;

			self.$headItems.eq(self.activeIndex).removeClass('active');
			self.$headItems.eq(idx).addClass('active');

			self.set();
			
			var distanceObj=self.calcDistance(idx);

			self.$head.velocity({ translateY: '+='+distanceObj.head },0);
			self.$body.velocity({ translateY: '+='+distanceObj.body },0);
			self.$headClone.css('margin-top',distanceObj.head+parseInt(self.$headClone.css('margin-top')));
			
			self.activeIndex=idx;

			self.assignClasses();


		},
		calcDistance:function(idx){
			var self=this;

			self.$headCloneItems.eq(self.activeIndex).removeClass('active');
			self.$headCloneItems.eq(idx).addClass('active');

			return {
				head:self.headCenterPos-self.$headCloneItems.eq(idx).offset().top,
				body:self.elemTopPos-self.$bodyItems.eq(idx).offset().top
				
			}

		},
		assignClasses:function(){
			var self=this;

			self.destroy();

			self.$headCloneItems.each(function(){
				var $this=$(this),
					idx=$this.index();

				if (idx==self.activeIndex) 
					return;

				
				var elTop=$this.offset().top;

				if ((elTop+self.headItemHeight<self.elemTopPos) || (elTop>self.elemTopPos+self.elemHeight)){
					$this.addClass('out-of-view');
					self.$headItems.eq(idx).addClass('out-of-view');
					return;
				}
				
				if ((elTop+self.headItemHeight<self.elemTopPos+self.headItemBoundry) || (elTop>self.elemTopPos+self.elemHeight-self.headItemBoundry)){
					$this.addClass('on-edge');
					self.$headItems.eq(idx).addClass('on-edge');
					return;
				}
				

				if ((elTop+self.headItemHeight<self.elemTopPos+self.headItemBoundry*2) || (elTop>self.elemTopPos+self.elemHeight-self.headItemBoundry*2)){
					$this.addClass('near-edge');
					self.$headItems.eq(idx).addClass('near-edge');
					return;
				}

				
			});
			

		},
		destroy:function(){
			var self=this;
			self.$headCloneItems.removeAttr('class');
			self.$headCloneItems.eq(self.activeIndex).addClass('active');
			self.$headItems.removeAttr('class');
			self.$headItems.eq(self.activeIndex).addClass('active');

		},
		bindUIActions:function(){
			var self=this;

			self.$headItems.on('click',function(){
				self.goTo($(this).index());
			});

			$(window).on('debouncedresize',function(){
				self.set();
			});
		}

	};
	//*********** Timeline Tabs Begin *******//
	
	
	//*********** Agenda Begin *******//
	var olAgenda={
		init:function($elem){
			this.$elem=$elem;

			if (this.$elem.hasClass('sticky-type')){
				this.stickyHead();
			}

			var $filters=this.$elem.find('ul.filters');
			if ($filters.length){
				this.setFilters($filters);
			}
			this.prepare();
			this.bindUIActions();
		},
		bindUIActions:function(){
			var self=this;

			self.$toggleableItems.on('click',function(){
				var $this=$(this),
					state='show';

				if ($this.hasClass('active')){
					state='hide';
				}
				$this.toggleClass('active');

				self.toggleIt($this.find('.extra-description'),state);
			});
		},
		prepare:function(){
			var self=this,
				$toggleElem=$('<div></div>').addClass('toggle-trigger');
			
			self.$toggleableItems=$();
			self.$allItems=self.$elem.find('.item');
			self.$allItems.each(function(){

				var $this=$(this);

				var $desc=$this.find('.extra-description');

				

				if ($desc.length){
					$this.addClass('toggleable');
					$this.append($toggleElem.clone());
					self.$toggleableItems=self.$toggleableItems.add($this);
				}
			});	
		},
		toggleIt:function($el,state){
			if (state=='show'){
				$el.slideDown();
			}else{
				$el.slideUp();
			}
			
		},
		stickyHead:function(){
			var self=this;

			self.$elem.find('.section-head .date').each(function(){
				var $this=$(this);

				$this.stick_in_parent({
					offset_top:window.olStickyOffset+10
				});
			});
		},
		setFilters:function($filters){
			var self=this,
				$lis=$filters.find('li');

			$filters.find('a').on('click',function(e){
				e.preventDefault();

				var $this=$(this),
					filter=$this.data('filter'),
					$li=$this.parent();

				if ($li.hasClass('active'))
					return false;

				$lis.removeClass('active');
				$li.addClass('active');
				
				if($this.data('filter') == "*"){
					self.$allItems.slideDown({ duration: 200, easing: "easeInOutQuart" });
				} else{
					self.$allItems.each(function(){
						var $this=$(this);

						if ($this.data('filter')==filter ){
							$this.slideDown({ duration: 200, easing: "easeInOutQuart" });
						}else{
							$this.slideUp({ duration: 200, easing: "easeInOutQuart" });
						}
					});
				}
					
			});
		}
	};
	//*********** Agenda End *******//


	//*********** Lightbox Begin *******//
	var lightBox={

		init:function(){
			var self=this,
				descMode;

			self.localvideo={
				autoPlay:false,
				preload:'metadata',
				webm :true,
				ogv:false	
			};

			$('.ol-lightbox').each(function(){
				var $this=$(this);
				if (!$this.parents('.ol-lightbox-gallery').length)
					self.singleBox($this);
			});
			

			$('.ol-lightbox-gallery').each(function(){
				self.galleryBox($(this));	
			});

			this.bindUIActions();
		
			
		},
		generateVideo:function(src,poster){
			var self=this;
			//here we generate video markup for html5 local video
			//We assumed that you have mp4 and webm or ogv format in samepath (video/01/01.mp4 & video/01/01.webm)
			var basePath=src.substr(0, src.lastIndexOf('.mp4'));
			var headOptions='';
			if (self.localvideo.autoPlay){
				headOptions+=' autoplay';
			}
			headOptions +='preload="'+self.localvideo.preload+'"';

			var markup='<video class="mejs-player popup-mejs video-html5" controls '+headOptions+' poster="'+poster+'">'+
				'<source src="'+src+'" type="video/mp4" />';

			if (self.localvideo.webm){
				markup+='<source src="'+basePath+'.webm" type="video/webm" />';
			}
	
			if (self.localvideo.ogv){
				markup+='<source src="'+basePath+'.ogv" type="video/ogg" />';
			}
			markup+='</video>'+'<div class="mfp-close"></div>';

			return markup;

		},
		generatedesc:function(descMode){
			var self=this,
				markup ='<div class="container">'+	
							'<div class="mfp-figure '+descMode+'">'+
								'<button title="Close (Esc)" type="button" class="mfp-close">x</button>'+
								'<figure>'+
									'<div class="wrapper">'+
										'<div class="mfp-description">'+
											'<figcaption class="mfp-figcaption">'+
													'<div class="mfp-title"></div>'+
													'<div class="mfp-desc"></div>'+
											'</figcaption>'+
										'</div>'+
										'<div class="mfp-content-container">'+
											'<div class="mfp-img"></div>'+
										'</div>'+
									'</div>'+
								'</figure>'+
							'</div>'+
						'</div>';
			return markup;
		},
		bindUIActions:function(){
			var self=this,
				$body=$('body');

			
			
			$('body').on('click','.mfp-container',function(e){
			if( e.target !== this ) 
				return;
			$(this).find('.mfp-close').trigger('click');
		});


		},
		singleBox:function($elem){
			var self=this;
			$elem.magnificPopup({
				type: 'image',
				closeOnContentClick: false,
				closeOnBgClick:false,
				mainClass: 'mfp-fade',
				iframe: {
					markup: '<div class="mfp-iframe-scaler">'+
				            '<div class="mfp-close"></div>'+
				            '<iframe class="mfp-iframe" frameborder="0" allowfullscreen></iframe>'+
				            '<div class="mfp-title"></div>'+
				          '</div>'
				},
				image:{
					verticalFit: true,
				},
				callbacks:{
					elementParse: function(item) {
						var popType=item.el.attr('data-type')||'image';
						// if (popType=='localvideo'){
						// 	item.type='inline';
						// 	var poster=item.el.attr('data-poster')||'';
						// 	item.src=self.generateVideo(item.src,poster);
						// }else{
							if(popType=='descriptive'){
								item.type='image';
								if(item.el.attr('data-type')=='descriptive'){
						    		descMode = 'with-desc';
									if(item.el.hasClass('horizontal')){
										descMode.concat('horizontal');
									}
									this.st.image.markup = self.generatedesc(descMode);
						    	}
							}else{
								item.type=popType;
							}
							
						// }
			    	},
	    			markupParse: function(template, values, item) {
	    				if(item.el.attr('title')){
	    					values.title = '<h3 class="title">'+item.el.attr('title')+'</h3>';
	    				}
	    				if(item.el.attr('desc')){
	    					values.desc = item.el.attr('desc');
	    				}
			    	},
			    	open: function() {
					// sideS.$exteras=$('.move-with-js').add('.mfp-wrap');
			  		// $('.popup-mejs').mediaelementplayer();
			  		}
				}
			});

		},
		galleryBox:function($elem){
			var self=this,
			$this=$elem,
			itemsArray=[];

			$elem.magnificPopup({
				delegate: '.ol-lightbox',
				closeOnBgClick:false,
				closeOnContentClick:false,
				removalDelay: 300,
				mainClass: 'mfp-fade',
				iframe: {
					markup: '<div class="mfp-iframe-scaler">'+
				            '<div class="mfp-close"></div>'+
				            '<iframe class="mfp-iframe" frameborder="0" allowfullscreen></iframe>'+
				            '<div class="mfp-title"></div>'+
				            '<div class="mfp-counter"></div>'+
				          '</div>'
				},
				gallery: {
				enabled: true,
				tPrev: 'Previous',
				   tNext: 'Next',
				   tCounter: '%curr% / %total%',
				   arrowMarkup: '<a class="tj-mp-action tj-mp-arrow-%dir% mfp-prevent-close" title="%title%"><i class="fa fa-angle-%dir%"></i></a>',
				},
				callbacks:{
					elementParse:function(item){
						
						var	popType=item.el.attr('data-type') || 'image',
							source=item.el.attr('href');

						

						// if (popType=='localvideo'){
						// 	item.src=self.generateVideo(source,item.el.attr('data-poster')||'');
						// 	item.type='inline';
						// }else{
							if(popType=='descriptive'){
								item.type='image';
								if(item.el.attr('data-type')=='descriptive'){
						    		descMode = 'with-desc';
									if(item.el.hasClass('horizontal')){
										descMode.concat('horizontal');
									}
									this.st.image.markup = self.generatedesc(descMode);
						    	}
							}else{
								item.type=popType;
							}
						// }
					},
					markupParse: function(template, values, item) {
	    				if(item.el.attr('title')){
	    					values.title = '<h3 class="title">'+item.el.attr('title')+'</h3>';
	    				}
	    				if(item.el.attr('desc')){
	    					values.desc = item.el.attr('desc');
	    				}
			    	},
					open:function(){
						// sideS.$exteras=$('.move-with-js').add('.mfp-wrap');
						// $('.popup-mejs').mediaelementplayer();
					},
					change: function() {
				        if (this.isOpen) {
				            this.wrap.addClass('mfp-open');
				        }
				        //$('.popup-mejs').mediaelementplayer();
				    }
				},
				type: 'image' // this is a default type
			});

			itemsArray=[];
		}
	};
	//*********** Lightbox End *******//


	//*********** Grid Handler Begin *******//
	var olGrid = {

		init: function($grid){
			this.$grid = $grid;
			this.$filtersWrapper = $('.ol-grid-filters');
			this.$defaultFilters=this.$filtersWrapper.find('.default-filters');
			this.$selectFiltersWrapper=this.$filtersWrapper.find('.select-filters');
			this.$selectFilters=this.$selectFiltersWrapper.find('select');
			this.$selectDummyValue=$('<span>').addClass('select-value').appendTo(this.$selectFiltersWrapper);
			this.gridFlg=this.$grid.hasClass('grid');
			
			this.prepare();
			this.prepareIsotope();
			this.bindUIActions();
		},
		bindUIActions:function(){
			var self=this,
				selector;
			 // click event in isotope filters
		    self.$defaultFilters.on('click', 'a',function(e){
				e.preventDefault();
				$(this).parent('li').addClass('active').siblings().removeClass('active');
				selector = $(this).attr('data-filter');
				self.filter(selector);
		    });

		    this.$selectFilters.on( 'change', function(){
		    	selector = this.value;
		    	self.$selectDummyValue.text(self.$defaultFiltersAnchors.filter(function(){
		    		return $(this).data('filter')==selector;
		    	}).text());
		    	self.filter(selector);
		    });
		    
		},
		filter: function(selector){
			var self=this;
			self.$grid.isotope({filter:selector})
		},
		prepareIsotope:function(){
			var self=this;

			if (self.gridFlg){
				self.isotopeGrid();
			}else{
				self.$grid.imagesLoaded( function() {
				  self.isotopeGrid();
				});
			}
		},
		isotopeGrid: function(){
			var widthClass=(this.$grid.find('.grid-sizer').length)?'.grid-sizer':'.grid-item';

			this.$grid.isotope({
				itemSelector: '.grid-item',
				percentPosition: true,
				masonry: {
				// use outer width of grid-sizer for columnWidth
					columnWidth: widthClass
				}
			})
		},
		prepare:function(){
			var self=this;
			var $filtersLi=self.$defaultFilters.children('li'),
				$activeLi=$filtersLi.filter('.active');

			$activeLi=($activeLi.length>0)?$activeLi:$filtersLi.first();
			self.$defaultFiltersAnchors=$filtersLi.find('a');
			self.$selectDummyValue.text($activeLi.text());
		}
	};
	//*********** Grid Handler End *******//
	

	//*********** Social Shares Begin *******//
	var socialShare = function(){
		var delayFactor = 60,
		deActivate = function($elem , direction){
			var $items = $elem.find(".items a");
			$elem.removeClass('active').addClass("in-active");
			if(direction== "bottom"){
				$items.each(function(indexInArray){
					$(this).animate({
						opacity: 0,
						bottom: "-=15"
					}, indexInArray*delayFactor, function(){
						$(this).fadeOut;
					});
				});
			}else{
				$items.each(function(indexInArray){
					$(this).animate({
						opacity: 0,
						top: "-=15"
					}, indexInArray*delayFactor, function(){
						$(this).fadeOut;
					});
				});
			}
		},
		activate = function($elem , direction){
			var $items = $elem.find(".items a");
			$elem.removeClass("in-active").addClass('active');
			if(direction== "bottom"){
				$items.each(function(indexInArray){
					$(this).fadeIn().animate({
						opacity: 1,
						bottom: 0
					}, indexInArray*delayFactor);
				});
			}else{
				$items.each(function(indexInArray){
					$(this).fadeIn().animate({
						opacity: 1,
						top: 0
					}, indexInArray*delayFactor);
				});
			}
		};
		$(".social-share").each(function(){
			var $elem = $(this),
				direction = "top";
			//deActive all the social-share items
			if ($elem.hasClass("bottom")){
				direction = "bottom";
				deActivate($elem, direction);
			}else{
				deActivate($elem, direction);
			}
			//handle clicks
			$elem.find(".trigger").click(function(){
				$elem = $(this).parent(".social-share");
				if ($elem.hasClass("in-active")){
					if ($elem.hasClass("bottom")){
						direction = "bottom";
						activate($elem, direction);
					}else{
						activate($elem, direction);
					}
				}else{
					if ($elem.hasClass("bottom")){
						direction = "bottom";
						deActivate($elem, direction);
					}else{
						deActivate($elem, direction);
					}
				};
			});
		});
	};
	//*********** Social Shares End *******//


	//*********** Sync columns height Begin *******//
	var syncHeight={
		init:function($refEl){
			this.$refEl=$refEl;
			this.fullWidthMargin=60;
			var $syncChilds=this.$refEl.find('.sync-me');

			if (!$syncChilds.length){
				$syncChilds=$refEl.children();
			}
			this.$elems=$syncChilds;

			if (!this.checkFullWidth()){
				this.sync();
			}
			this.bindUIActions();
		},
		sync:function(){
			this.$elems.outerHeight(this.$refEl.outerHeight());
		},
		destroy:function($elems){
			$elems.css('height','');
		},
		bindUIActions:function(){
			var self=this;

			$(window).on('debouncedresize',function(){
				self.destroy(self.$elems);
				if (!self.checkFullWidth()){
					self.sync();
				}
			});
		},
		checkFullWidth:function(){
			var self=this;
			return (self.$refEl.outerWidth()-self.$elems.first().outerWidth()<=self.fullWidthMargin);
		}
	};
	//*********** Sync columns height End *******//



	//*********** Extendable backgrounds Begin *******//
	var extendBg={
		init:function($wrapper){
			this.$wrapper=$wrapper;
			this.$extendableElem=$wrapper.find('.extend-left,.extend-right');
			this.$targetCols=this.$extendableElem.parent();
			this.$columns=$wrapper.children();
			this.fullWidthMargin=30;


			this.extendCore();
			this.bindUIActions();
		},
		extendCore:function(){
			
			if(this.checkFullWidth()){
				this.destroy();
			}else{
				syncHeight.init(this.$wrapper);
				this.extendWidth(this.$extendableElem);	
			}
		},
		extendWidth:function($el){
			var self=this,
				elWidth=$el.css('width','').width(),
				sideMargin;

			if ($el.hasClass('.extend-right')){
				//Extend it to the right side of window
				sideMargin=$(window).width()-($el.offset().left+elWidth);
			}else{
				//Extend it to the left side of window
				sideMargin=$el.offset().left;
			}

			$el.width(elWidth+sideMargin);

		},
		destroy:function(){
			syncHeight.destroy(this.$columns);
			setBg.destroy(this.$wrapper.find('.set-bg'));
			
			this.$extendableElem.css('width','');
			this.$wrapper.addClass('extend-destroy');
		},
		checkFullWidth:function(){
			var self=this;
			return (self.$wrapper.width()-self.$targetCols.first().width()<=self.fullWidthMargin);
		},
		bindUIActions:function(){
			var self=this;

			$(window).on('debouncedresize',function(){
		        if(self.checkFullWidth()){
					//extendable column is fullwidth so we should destroy the whole thing
					self.destroy();

				}else{

					//Redo the math
					self.$wrapper.removeClass('extend-destroy');
					self.extendCore();
					setBg.init(self.$wrapper.find('.set-bg'));
					//self.$wrapper.find('.owl-videobg').owlVideoBg('update');
				}
		    });
				
		}
	};
	//*********** Extendable backgrounds Begin *******//


	//*********** Retina Images Handler Begin *******//
	var retina={
		init:function($elem){
			this.$elem=$elem;
			this.retinaSuffix="@2x";
			
			if (!isRetinaDisplay()) return false;
			var self=this,
				imgSrc=$elem.attr('src');

			if (!imgSrc) return false;

			//Generate retina image path based on the suffix
			var retinaSrc=imgSrc.replace(/\.(?!.*\.)/, self.retinaSuffix +".");

			//check if there is any retina verison of the image 
			this.preload(retinaSrc,function(retinaImg){
				if(retinaImg){
					self.setRetina(retinaImg);
				}else{
					console.warn('Error loading the retina image');
					return false;
				}
			});
		},
		preload:function(imgSrc,callback){
			var img = new Image();
			img.src = imgSrc;
			img.onerror=function(){
				return callback(false);
			};
			img.onload = function() {
				return callback(img);
			};
		},
		setRetina:function(retinaImg){
			var self=this;

			self.$elem.attr('src',retinaImg.src);
			var noDimensionFlg=isNaN(parseInt(self.$elem.attr('width')))&&isNaN(parseInt(self.$elem.attr('height')));
			if (noDimensionFlg){
				self.$elem.attr('width',retinaImg.width/2);
				self.$elem.attr('height',retinaImg.height/2);
			}
		}
	};
	//*********** Retina Images Handler End *******//

	

	//*********** Cover Images in a Container Begin *******//
	var imageFill={
			
	  init:function($container,callback){
	    this.container=$container;
	    this.setCss(callback);
	    this.bindUIActions();

	  },
	  setCss:function(callback){
	    $container=this.container;
	    $container.imagesLoaded(function(){
	      var containerWidth=$container.width(),
	        containerHeight=$container.height(),
	        containerRatio=containerWidth/containerHeight,
	        imgRatio;

	      $container.find('img').each(function(){
	        var img=$(this);
	        imgRatio=img.width()/img.height();
	        
	        if (img.css('position')=='static'){
	        	img.css('position','relative');
	        };
	        if (containerRatio < imgRatio) {
	          // taller
	          img.css({
	              width: 'auto',
	              height: containerHeight,
	              top:0,
	              left:-(containerHeight*imgRatio-containerWidth)/2
	            });
	        } else {
	          // wider
	          img.css({
	              width: containerWidth,
	              height: 'auto',
	              top:-(containerWidth/imgRatio-containerHeight)/2,
	              left:0
	            });
	        };
	      });
	      if (typeof(callback) == 'function'){
	      	callback();
	      };
	    });
	  },
	  bindUIActions:function(){
	    var self=this;
	    $(window).on('debouncedresize',function(){
	        self.setCss();
	    });
	  }
	};
	//*********** Cover Images in a Container End *******//



	//*********** Search Handler Begin *******//
	var searchHandler={
		init:function($elem){
			this.$elem=$elem;
			this.$searchArea=$('.search-area');
			this.customAnimFlg=this.$searchArea.hasClass('fullscreen');

			this.bindUIActions();
			
		},
		bindUIActions:function(){
			var self=this;

			self.$elem.children('a').on('click', function(e){
				e.preventDefault();
				self.displayArea('show');
			});

			self.$searchArea.find('.close-btn').on('click',function(e){
				e.preventDefault();
				self.displayArea('hide');
			});

			$(document).keyup(function(e) {
				if (e.keyCode == 27) {
					if (self.$searchArea.hasClass("is-visible")) self.displayArea('hide');
				}
			});
		},
		displayArea:function(mode){
			var self=this;

			if (mode=='show'){
				self.$searchArea.toggleClass('is-visible');
			}else{
				self.$searchArea.removeClass('is-visible');
			}

			self.animateArea(mode);
		},
		animateArea:function(mode){
			var self=this;

			if (!self.customAnimFlg) return;

			if (mode=='show'){
				self.$searchArea.velocity({opacity:1,top:0},{display:"block",duration:200});
			}else{
				self.$searchArea.velocity({opacity:0,top:-150},{display:"none",duration:5});
			}
		}
	};
	//*********** Search Handler End *******//



	//*********** Reveal Animation on scroll Begin *******//
	var revealAnimate={
		init:function($elem){
			this.$elem=$elem;
			this.disableMobile=true;
			this.offset=$elem.data('offset')?$elem.data('offset'):-50;
			this.animatedFlag=false;

			this.animOptions={
				duration:$elem.data('duration'),
				delay:$elem.data('delay'),
				iteration:$elem.data('iteration')
			};
			
			this.prepare();
			this.checkInView();
			this.bindUIActions();


		},
		prepare:function(){
			var self=this;
			var cssObj={};

			self.animOptions.name=self.$elem.css('animation-name');

			for (var option in self.animOptions){
				if (self.animOptions[option]){
					cssObj['animation-' + option]=self.animOptions[option];
					cssObj['-webkit-animation-' + option]=self.animOptions[option];
					self.$elem.css(cssObj);
				}
			}

			self.$elem.css('visibility','hidden').css('animation-name','none');
			
		},
		bindUIActions:function(){
			var self=this;
			$.og.$window.on('scroll', self.checkInView.bind(self));
		},
		checkInView:function(){
			var self=this;
			if (self.animatedFlag) return false;

			if (verge.inY(self.$elem, self.offset)){
				//$.og.$window.off('scroll');
				self.animatedFlag=true;
				self.$elem.css('animation-name',self.animOptions.name);

				if (self.animOptions.delay){
					setTimeout(function(){
						self.setAnimations();
					},parseFloat(self.animOptions.delay)*1000);
				}else{
					self.setAnimations();
				}
				
			}
		},
		setAnimations:function(){
			this.$elem.css('visibility','visible').addClass('animated');
		}
	};
	//*********** Reveal Animation on scroll End *******//


	//*********** Parallax Backgrounds Begin *********//

	var parallaxer = {
		settings:function(){
			return{
				// zoom in move
				"mode-1" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1);",
					'end' 	: "transform:translate3d(0px, %distance%px, 0.1px) scale(1.4);",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'distance'
				},
				// zoom out move
				"mode-2" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4);",
					'end' 	: "transform:translate3d(0px, %distance%px, 0.1px) scale(1);",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'distance'
				},
				// zoom in
				"mode-3" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1);",
					'end' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4);",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'none'
				},
				//zoom out
				"mode-4" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4);",
					'end' : "transform:translate3d(0px, 0px, 0.1px) scale(1);",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'none'
				},
				// zoom in opacity up
				"mode-5" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1); opacity:0;",
					'end' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4); opacity:2",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'none'
				},

				// zoom out opacity down
				"mode-6" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1); opacity:2;",
					'end' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4); opacity:-1",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'none'
				},
				"mode-title" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px) scale(1); opacity:2;",
					'end' : "transform:translate3d(0px, 0px, 0.1px) scale(1.4); opacity:-1",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'none'
				},
				"mode-header-content" : {
					'start' : "transform:translate3d(0px, 0px, 0.1px);opacity:1",
					'end':  "transform:translate3d(0px, %height%px, 0.1px);opacity:-0.5",
					'data-anchor-target' : ".page-head", 
					'type' :'height'
				},
				"mode-header-demo-2":{
					'start' : "transform:translate3d(0px, 0px, 0.1px);opacity:1",
					'end':  "transform:translate3d(0px, %height%px, 0.1px);opacity:-1",
					'data-anchor-target' : ".page-head", 
					'type' :'height'
				},
				"default":{
					'start' : "transform:translate3d(0px, 0px, 0.1px);",
					'end' : "transform:translate3d(0px, %distance%px, 0.1px);",
					'data-anchor-target' : "."+this.$elem.UniqueClass, 
					'type' :'distance'
				}
			}
		},
		init:function($elem){
			this.$elem = $elem;
			this.$elem.readyState=false;
			this.attsObj = {};
			this.parallaxMode=this.$elem.data('parallax-mode')?this.$elem.data('parallax-mode'):'default';
			
			this.setAttributes();

		
			//wait for the prepare to be done.
			var self = this;

			this.prepare(function(){
				self.$elem.readyState=true;
				self.$elem.trigger('parallaxReady');
				self.$layer.addClass("parallax-"+self.parallaxMode);
				
				self.getAnimations();
				self.setAnimations();
				

			});

			this.bindUIActions();
			
		},
		setAttributes:function(){
			this.elementOffsetTop=this.$elem.offset().top;
			this.elemHeight=this.$elem.outerHeight();
			this.elemWidth=this.$elem.outerWidth();
			this.windowHeight=$.og.$window.height();
		},
		bindUIActions:function(){
			var self=this;
			$.og.$window.on('olRevsReady',function(){
				if (self.$elem.readyState){
					self.manualRevUpdate();
				}else{
					self.$elem.on('parallaxReady',function(){
						self.manualRevUpdate();
					});
				}
				
			});
		},
		manualRevUpdate:function(){
			var self=this;
			
			self.destroyAnimations();
			self.setAttributes();
			self.getAnimations();
			self.setAnimations();
			if (checkParallaxesState()){
				skrollrHandler.updateSkrollr();
				$('.rev_slider').revredraw();
			}
		},
		prepare:function(callback){
			var self=this;

			var prefix = "ol-para-bg-";
			self.$elem.removeClassPrefix(prefix);
			self.$elem.UniqueClass = prefix+ makeid();
			this.$elem.addClass(this.$elem.UniqueClass);

			// get imagesrc
			var imgSrc = self.$elem.data('img-src');
			if ( ! imgSrc ) {
				self.$layer=self.$elem;
				callback();
				return false;
			}

			// create imageLayer
			self.$imageLayer =$('<div></div>').addClass('parallax-bg-elem');
			
			// add image to it's background
			self.$imageLayer.css('background-image','url('+imgSrc+')');
			

			//append imageLayer to the container
			self.$elem.append(self.$imageLayer);

			// get image size
			self.getImageSize(imgSrc,function(imgDimesnion){


				//set bg element's height and enshure that it is bigger than the container so can be parallaxed
				self.$imageLayer.height = self.elemWidth / (imgDimesnion.width/imgDimesnion.height);
				self.$imageLayer.height = Math.max( self.elemHeight*1.5 , self.$imageLayer.height);
				
				// set height 
				self.$imageLayer.css({
					height 	: self.$imageLayer.height,
					// in case we want to center align
					top :  -(self.$imageLayer.height - self.elemHeight)/2
				});

				//self.$elem.append(self.$imageLayer);

				self.$layer=self.$imageLayer;
				// we are done!
				callback();

			});
		},
		getAnimations:function(){
			var self = this;

			self.attsObj = self.settings()[self.parallaxMode];
			self.attsObj=self.assignVariables(self.attsObj);

			//Detect start and end range
			self.attsObj=self.assignRange(self.attsObj);			
		},
		setAnimations:function(){
			var self = this;
			self.$layer.attr(self.attsObj);
		},
		destroyAnimations:function(){
			var self=this;
			self.removeDataAttributes(self.$layer);
		},
		getImageSize:function(imgSrc,callback){
			var img = new Image();
			img.src = imgSrc;
			img.onload = function() {
				var imgDimesnion={
					width:img.width,
					height:img.height
				};
				callback(imgDimesnion);
			}	
		},
		assignRange:function(attsObj){
			var self=this,
				start,end,obj=attsObj;

			if (self.elementOffsetTop>self.windowHeight/2){
				start='data-bottom-top';
				end='data-top-bottom'
			}else{
				start='data-top-top';
				end='data-top-bottom'
			}

			if (attsObj['start']){
				obj[start]=attsObj['start'];
				delete obj['start'];
			}

			if (attsObj['end']){
				obj[end]=attsObj['end'];
				delete obj['end'];
			}
			return obj;
		},
		calcDistance:function(){
			var self=this,
				distance;

			if (self.elementOffsetTop>self.windowHeight){
				distance=self.windowHeight+self.elemHeight;
			}else{
				distance=self.elemHeight+self.elementOffsetTop;
			}
			var maxRatio=(self.$layer.height-self.elemHeight)/(2*distance);
			self.elemBgRatio=self.$elem.data('bg-parallax-factor')?self.$elem.data('bg-parallax-factor'):Math.min(maxRatio,0.6);

			var ratio=Math.max(Math.min(Math.abs(self.elemBgRatio), maxRatio),0.05);

			return distance*ratio*Math.sign(self.elemBgRatio);
		},
		assignVariables:function(attsObj){
			var self=this;

			if (attsObj['type']=='height'){
				attsObj['end']=attsObj['end'].replace(/%\w+%/g, parseInt($('.page-head').height()/2));
			}else if (attsObj['type']=='distance'){
				attsObj['end']=attsObj['end'].replace(/%\w+%/g, parseInt(self.calcDistance()));
			}

			return attsObj;
		},
		removeDataAttributes:function($target) {
			var i,
			attrName,
			dataAttrsToDelete = [],
			dataAttrs = $target.get(0).attributes,
			dataAttrsLen = dataAttrs.length;

			for (i=0; i<dataAttrsLen; i++) {
				if ( 'data-' === dataAttrs[i].name.substring(0,5) ) {
					dataAttrsToDelete.push(dataAttrs[i].name);
				}
			}
			$.each( dataAttrsToDelete, function( index, attrName ) {
				// remove attr from element
				$target.removeAttr( attrName );
				// remove data
				$target.removeData(attrName.substr(5));
			});
		}
	};

	//*********** parallax Backgrounds End *********//

	//*********** Skrollr Parallax Begin *******//
	var skrollrHandler={
		init:function(){
			this.skrollrFlg=false;

			if (olIsTouchDevice()) return false;

			this.makeDecision();
			this.bindUIActions();
		},
		bindUIActions:function(){
			var self=this;
			$(window).on('debouncedresize',function(){
				self.makeDecision();
			});
		},
		makeDecision:function(){
			var self=this;
			if ($(window).width() > 767){
				if (!self.skrollrFlg){
					self.initSkrollr();
					self.skrollrFlg=true;
				}
			}else{
				self.destroy();
			}
		},
		initSkrollr:function(){
			skrollr.init({
				forceHeight:false
			});
		},
		destroy:function(){
			
			skrollr.get() && skrollr.get().destroy();
			self.skrollrFlg=false;
		},
		updateSkrollr:function(){
			var self=this;
			if (self.skrollrFlg){
				self.destroy();
				self.initSkrollr();
			}
		}
	};
	//*********** Skrollr Parallax End *******//


	//*********** Slider Revolution manual ready event *******//
	var olRevSliderHandler={
		init:function(){
			this.$revSliders=$('.rev_slider_wrapper');
			this.sliderNum=this.$revSliders.length;
			if (!this.sliderNum>0) return;

			this.checkSliders();
		},
		checkSliders:function(){
			var self=this,
				counter=0;
			self.$revSliders.each(function(){
				var $this=$(this);

				$this.bind('revolution.slide.onloaded', function() {
					counter++;
					if (counter==self.sliderNum){
						
						self.slidersDone();
					}

				});
			});
		},
		slidersDone:function(){
			$.og.$window.trigger('olRevsReady');
		}
	};
	//*********** Slider Revolution manual ready event *******//


	//*********** Handle Hovers on touch Begin *******//
	var olHoverHandler={
		init:function(){
			if (!$.og.isTouchDevice) return false;
			this.$wrapper=$('#wrapper');
			this.selector='.ol-hover';

			this.bindUIActions();
		},
		bindUIActions:function(){
			var self=this;
			self.$wrapper.on('click',self.selector,function(){

				var $this=$(this),
					elemTag=$this[0].tagName.toLowerCase();

				if (this.elemTag!='a'){
					$this.addClass('touch-hover');
					$.og.$body.one('click',function(){
						$this.removeClass('touch-hover');
					});
				}
			});
		}
	};
	//*********** Handle Hovers on touch End *******//


	//*********** Background Videos Begin*********//
	function videobg(){
		$('.owl-videobg').owlVideoBg({    
	    	autoGenerate:{
	    		posterImageFormat:'png'
	    	},
	    	preload:'auto'
		    	
	    });
	};
	//*********** Background Videos End*********//


	//*********** Sticky Header Offset Begin *********//
	function setStickyOffset(){
		var offset=0;
		//calc offset top for fixing head items
		if ($("body").hasClass('sticky-header')){
			var $header=$('#header');


			$header = $header.addClass("is-sticky no-transition");
			offset+=$header.outerHeight();
			$header.removeClass('is-sticky no-transition');
		}
		
		window.olStickyOffset=offset;
	};
	//*********** Sticky Header Offset End *********//

	//check if all parallaxes are done
	function checkParallaxesState(){

		if (window.olParallaxController == undefined){
			window.olParallaxController={};
			window.olParallaxController.ready=false;
			window.olParallaxController.num=$('.parallax-layer').length;
			window.olParallaxController.progress=1;
		}
		window.olParallaxController.progress++;
		if ( window.olParallaxController.progress == window.olParallaxController.num){
			window.olParallaxController.ready=true;
			return true
		}
		return false;
	};
	
	
	//*********** Init Handler for all Functions *******//
	var initRequired={
		init:function(){
			$.og={
				$body:$('body'),
				$header:$('#header'),
				$window:$(window),
				isTouchDevice:olIsTouchDevice()
			};

			this.runMethods();
			this.runInlines();
		},
		runMethods:function(){
			//initialize all required functions here

			createObjInstance('#primary-menu',olmenu);
			searchHandler.init($('.ol-search-trigger'));
			setStickyOffset();
			logoHandler.init();
			socialShare();
			lightBox.init();
			olHoverHandler.init();

			
			if ($.og.$body.hasClass("sticky-header"))
				stickyHeader.init();


			// Initialize plugins that can have multiple instance on same page	
			createObjInstance('img.ol-retina',retina);
			createObjInstance('.extend-bg-wrapper',extendBg);
			createObjInstance('.sync-cols-height',syncHeight);
			createObjInstance('.set-bg',setBg);
			createObjInstance('.ol-particles',particles);
			createObjInstance('.ol-timeline-tab',olTimeTab);
			createObjInstance('.ol-timeline.scrollable-timeline',olTimeline);
			createObjInstance('.parallax-layer',parallaxer);
			createObjInstance('.ol-agenda',olAgenda);
			createObjInstance('.ol-accordion',accordion);
			createObjInstance('.ol-side-navigation',accordion,{
				itemSelector:'li',
				headSelector:'a',
				bodySelector:'.sub-menu',
				activeClass:'active',
				initActiveClass:'current-menu-parent',
				toggleElemClass : ".menu-item-has-children",
				toggleEl:'<span class="ol-toggle">'
			});
			createObjInstance('.ol-animate',revealAnimate);
			createObjInstance('.ol-tab',tabs);
			createObjInstance('.ol-grid',olGrid);

			videobg();
			olRevSliderHandler.init();
		},
		runInlines:function(){
			//Run Inline functions Here


			//*********** Remove title from nav links *********//
			$("#header #nav a").attr('title','');

			//*********** Tooltips *********//
			$('[data-toggle="tooltip"]').tooltip();


			//*********** Google Map *********//
			var $gmap = $("#gmap , .gmap"); 
			if ($gmap.length > 0){
				$gmap.each(function(){
					var $gmap=$(this);
					var address= $gmap.attr('data-address') || 'Footscray VIC 3011 Australia';

					$gmap.gmap3({
					  map: {
					      options: {
					          maxZoom:15,
					          disableDefaultUI: true
					      }
					  },
					  styledmaptype: {
					      id: "mystyle",
					      options: {
					          name: "Style 1"
					      },
					      styles: [
					          {
					              featureType: "all",
					              stylers: [
					                  {"saturation": -100}, {"gamma": 0.9}
					              ]
					          }
					      ]
					  },
					  overlay:{
					    //Edit following line and enter your own address
					    address: address,
					    options:{
					      content: '<div id="map-marker"></div>',
					      offset:{
					        y:-100,
					        x: -25
					      }
					    }
					    },//Following maxZoom option is for setting the initial zoom of google map
						  autofit:{maxZoom: 15}
						},"autofit");

					$gmap.gmap3('get').setMapTypeId("mystyle");
				});
			}

			//*********** Search Filter *********//
			$(".search-box").each(function(){
				var $this = $(this);
				var $filters=$this.children(".filters");
				var $toggle = $this.find(".toggle-filter").first();

				$toggle.on('click',function(e){

					e.preventDefault();
					$this.toggleClass('fill-it');
					$filters.slideToggle();
				});

			});

			
			//*********** Type Writter *********//
			$(".ol-text-rotate").each(function(){
				var $this = $(this);
				var sentences = $this.data('words') || {};
				var arrayOfSentences = $.map(sentences, function(value, index) {
					return [value];
				});
				
				$(this).typed({
					strings: arrayOfSentences,
					typeSpeed: 100,
					backDelay: 1000,
					loop: true,
				});

			});

			//*********** Selectize and Datepicker *********//
			
			var $selectized = $(".selectize").selectize();

			var $datepickers = $(".pickdate").pickadate();

			$('.clear-selectize').on('click',function(e){
				
				e.preventDefault();
				
				$datepickers.each(function(i){
					var picker = $(this).pickadate('picker').clear();
				});
				$selectized.each(function(i){
					$selectized[i].selectize.clear();
				});

				//clear all inputs 
				$('.filters').find(":input").val('');
				
			});

			//*********** carousel *********//
			$(".owl-carousel.items").each(function() {
				var $this = $(this);
				
				var $cols_xxs = $this.data('cols-xxs') 	|| 1;
				var $cols_xs  = $this.data('cols-xs')  	|| 1;
				var $cols_sm  = $this.data('cols-sm')  	|| 2;
				var $cols_md  = $this.data('cols-md')  	|| 3;
				var $cols_lg  = $this.data('cols') 	|| 4;
				var $cols_lg  = $this.data('cols-lg')  	|| $cols_lg;

				if( $this.data('cols-all') ){
					$cols_lg = $cols_md = $cols_sm = $cols_xs = $cols_xxs = $this.data('cols-all');
				}
				
				
				$this.owlCarousel({
				 	items : $cols_lg,
				 	responsive : {
				 	    0 : {
				 	        items: $cols_xxs
				 	    },
				 	    480 : {
				 	        items: $cols_xs
				 	    },
				 	    768 : {
				 	         items: $cols_sm
				 	     },
				 	    992 : {
				 	        items: $cols_md,
				 	    },
				 	    1200 : {
				 	    	items: $cols_lg
				 	    }
				 	},
					autoplay : true,
					dots : $this.data('dots') || false,
					nav : $this.data('nav') || false,
					mouseDrag : true,
				 	stopOnHover : true,
				 	slideSpeed : $this.data('slidespeed') || 2000,
				 	paginationSpeed : $this.data('paginationspeed') || 2000,
				 	rewindSpeed : $this.data('rewindspeed') || 1100,
				 	margin: $this.data('margin') || 0,
				 	callbacks: true,
				 	autoplayHoverPause: true,
				 	autoplayTimeout: $this.data('autoplaytime') || 3000,
				 	loop: $this.data('loop') || false,
				 });
			});

			//*********** countdown *********//
			$('.ol-countdown').each(function(){
				var $this = $(this),
				finalDate = $this.data('countdown');

				$this.countdown(finalDate, function(event) {
					var $this = $(this).html(event.strftime(''
					+ '<div>%w<span>weeks</span></div>'
					+ '<div>%d<span>days</span></div>'
					+ '<div>%H<span>hours</span></div>'
					+ '<div>%M<span>minuets</span></div>'
					+ '<div>%S<span>seconds</span></div>'));
				});
			});

		}
	};
	//*********** Init Handler for all Functions *******//


	//Run methods on DOM Ready
	$(document).ready(function(){
		initRequired.init();
	});

	//Run methods on Window load
	$(window).on('load',function(){
		skrollrHandler.init();
	});
	
})(jQuery);



//*********** Utility Functions Begin *******//

//Object Create function
if ( typeof Object.create !== 'function'  ){ // browser dose not support Object.create
    Object.create = function (obj){
        function F(){};
        F.prototype = obj;
        return new F();
    };
};

//Create Object Instance
function createObjInstance(selector,objName,options){
	$(selector).each(function(){
		var obj = Object.create( objName ); 
        	obj.init($(this),options);
        	
	});	
}

//Generates Random Text
function makeid(){
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for( var i=0; i < 5; i++ )
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

//Check for Retina devices
function isRetinaDisplay() {
    if (window.matchMedia) {
        var mq = window.matchMedia("only screen and (min--moz-device-pixel-ratio: 1.3), only screen and (-o-min-device-pixel-ratio: 2.6/2), only screen and (-webkit-min-device-pixel-ratio: 1.3), only screen  and (min-device-pixel-ratio: 1.3), only screen and (min-resolution: 1.3dppx)");
        return (mq && mq.matches || (window.devicePixelRatio > 1)); 
    }
}

//Check for Touch devices
function olIsTouchDevice(){
	var agent = navigator.userAgent.toLowerCase(),
		isChromeDesktop = (agent.indexOf('chrome') > -1 && ((agent.indexOf('windows') > -1) || (agent.indexOf('macintosh') > -1) || (agent.indexOf('linux') > -1)) && agent.indexOf('mobile') < 0 && agent.indexOf('android') < 0);
	return ('ontouchstart' in window && !isChromeDesktop);
}

//Ajax Contact Form
(function ($, window, document, undefined) {
    'use strict';

    var $form = $('#contact-form');

    $form.submit(function (e) {

    	e.preventDefault();

        // remove the error class
        $form.find('.ajax-results').remove();

        // get the form data
        var formData = {
            'name' : $('input[name="name"]').val(),
            'email' : $('input[name="email"]').val(),
            'subject' : $('input[name="subject"]').val(),
            'message' : $('textarea[name="message"]').val()
        };

        var actionUrl = $form.attr('action');

        // process the form
        $.ajax({
            type : 'POST',
            url  : actionUrl,
            data : formData,
            dataType : 'json',
            encode : true
        }).done(function (data) {
            // handle errors
            if (!data.success) {
            	var html = '<div class="ajax-results alert alert-danger alert-thin with-big-icons"><i class="oli oli-cancel"></i>';

                if (data.errors.name) {
                	html = html + data.errors.name + "</br>";
                }

                if (data.errors.email) {
                	html = html + data.errors.email+ "</br>";
                }

                if (data.errors.subject) {
                	html = html + data.errors.subject+ "</br>";
                }

                if (data.errors.message) {
                	html = html + data.errors.message+ "</br>";
                }

                html = html +  "</div>";

                $form.prepend(html);

            } else {
                // clean it up
                $form.find('input[type="text"],input[type="email"], textarea').val('');
                // dispaly message
                $form.prepend('<div class="alert alert-success alert-thin with-big-icons"><i class="oli oli-ok"></i>' + data.message + '</div>');
            }
        }).fail(function (data) {
            // for debug
            console.log(data);
            $form.prepend('<div class="alert alert-danger alert-thin with-big-icons"><i class="oli oli-cancel"></i>There was an error! Could not send email.</div>');
        });

        
    });
}(jQuery, window, document));

// Add remove class with prefix to jQuery
jQuery.fn.removeClassPrefix = function(prefix) {
	this.each(function(i, el) {
		var classes = el.className.split(" ").filter(function(c) {
			return c.lastIndexOf(prefix, 0) !== 0;
		});
		el.className = $.trim(classes.join(" "));
	});
	return this;
};

//*********** Utility Functions End *******//

function getAllMethods(object) {
    return Object.getOwnPropertyNames(object).filter(function(property) {
        return typeof object[property] == 'function';
    });
}

